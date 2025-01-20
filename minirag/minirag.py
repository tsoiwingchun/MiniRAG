import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

from .llm import *

from .operate import (
    chunking_by_token_size,
    extract_entities,
    local_query,
    global_query,
    hybrid_query,
    minirag_query,
    naive_query,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)

from .kg.neo4j_impl import Neo4JStorage

from .kg.oracle_impl import OracleKVStorage, OracleGraphStorage, OracleVectorDBStorage

# future KG integrations

# from .kg.ArangoDB_impl import (
#     GraphStorage as ArangoDBStorage
# )


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that an asyncio event loop is always available.

    This function attempts to retrieve the current event loop using 
    `asyncio.get_event_loop()`. If no event loop is found (which raises a 
    RuntimeError), it creates a new event loop, sets it as the current event 
    loop, and returns it.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        return asyncio.get_event_loop()

    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        return loop


@dataclass
class MiniRAG:
    working_dir: str = field(
        default_factory=lambda: f"./minirag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    
    # RAGmode: str = 'minirag'

    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = hf_model_complete#gpt_4o_mini_complete  # 
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  #'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        """
        Post-initialization method for setting up the MiniRAG instance.

        This method performs the following tasks:
        1. Initializes the logger and sets the logging level.
        2. Logs the initialization parameters.
        3. Sets up storage classes for key-value, vector, and graph storage.
        4. Creates the working directory if it does not exist.
        5. Initializes the LLM response cache if enabled.
        6. Limits the asynchronous function calls for the embedding function.
        7. Initializes storage for full documents, text chunks, and chunk entity relation graph.
        8. Initializes vector databases for entities, entity names, relationships, and chunks.
        9. Limits the asynchronous function calls for the LLM model function.

        Attributes:
            log_file (str): Path to the log file.
            key_string_value_json_storage_cls (Type[BaseKVStorage]): Class for key-value storage.
            vector_db_storage_cls (Type[BaseVectorStorage]): Class for vector database storage.
            graph_storage_cls (Type[BaseGraphStorage]): Class for graph storage.
            llm_response_cache (Optional[BaseKVStorage]): Cache for LLM responses.
            embedding_func (Callable): Embedding function with limited async calls.
            full_docs (BaseKVStorage): Storage for full documents.
            text_chunks (BaseKVStorage): Storage for text chunks.
            chunk_entity_relation_graph (BaseGraphStorage): Storage for chunk entity relation graph.
            entities_vdb (BaseVectorStorage): Vector database for entities.
            entity_name_vdb (BaseVectorStorage): Vector database for entity names.
            relationships_vdb (BaseVectorStorage): Vector database for relationships.
            chunks_vdb (BaseVectorStorage): Vector database for chunks.
            llm_model_func (Callable): LLM model function with limited async calls.
        """
        log_file = os.path.join(self.working_dir, "minirag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"MiniRAG init with param:\n  {_print_config}\n")

        # @TODO: should move all storage setup here to leverage initial start params attached to self.

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        ####
        # add embedding func by walter
        ####
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        ####
        # add embedding func by walter over
        ####

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        global_config=asdict(self)

        self.entity_name_vdb = (
            self.vector_db_storage_cls(
                namespace="entities_name",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"}
            )
        )

        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }

    def insert(self, string_or_strings):
        """
        Inserts a string or a list of strings into the database asynchronously.

        Args:
            string_or_strings (str or list of str): The string or list of strings to be inserted.

        Returns:
            The result of the asynchronous insertion operation.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        """
        Asynchronously inserts new documents and their corresponding chunks into storage.

        Args:
            string_or_strings (Union[str, List[str]]): A single string or a list of strings representing the documents to be inserted.

        Returns:
            None

        Raises:
            None

        This method performs the following steps:
            1. Converts a single string input into a list of strings if necessary.
            2. Computes unique document IDs and filters out documents that are already in storage.
            3. Logs the number of new documents to be inserted.
            4. Chunks the documents based on token size and computes unique chunk IDs.
            5. Filters out chunks that are already in storage.
            6. Logs the number of new chunks to be inserted.
            7. Inserts the new chunks into the chunk storage.
            8. Extracts entities and relationships from the new chunks and updates the knowledge graph.
            9. Inserts the new documents and chunks into their respective storages.
            10. Calls a finalization method if any new documents or chunks were inserted.

        Note:
            - The method uses asynchronous operations for database interactions.
            - Logging is used to provide information about the insertion process.
        """
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                entity_name_vdb=self.entity_name_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        """
        Asynchronously calls the `index_done_callback` method on each non-None storage instance.

        This method iterates over a list of storage instances and, for each instance that is not None,
        it appends the `index_done_callback` coroutine to a list of tasks. It then awaits the completion
        of all these tasks concurrently using `asyncio.gather`.

        The storage instances include:
        - self.full_docs
        - self.text_chunks
        - self.llm_response_cache
        - self.entities_vdb
        - self.entity_name_vdb
        - self.relationships_vdb
        - self.chunks_vdb
        - self.chunk_entity_relation_graph

        Returns:
            None
        """
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.entity_name_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query(self, query: str, param: QueryParam = QueryParam()):
        """
        Executes a synchronous query using the provided query string and query parameters.

        Args:
            query (str): The query string to be executed.
            param (QueryParam, optional): An instance of QueryParam containing query parameters. Defaults to an empty QueryParam instance.

        Returns:
            Any: The result of the asynchronous query execution.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        """
        Asynchronously performs a query based on the specified mode.

        Args:
            query (str): The query string to be processed.
            param (QueryParam, optional): The parameters for the query, including the mode. Defaults to QueryParam().

        Returns:
            The response from the query, which varies based on the mode.

        Raises:
            ValueError: If the mode specified in param is unknown.

        Modes:
            - "light": Uses the hybrid_query function.
            - "mini": Uses the minirag_query function.
            - "naive": Uses the naive_query function.
        """
        if param.mode == "light":
            response = await hybrid_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "mini":
            response = await minirag_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.entity_name_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                self.embedding_func,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def _query_done(self):
        """
        Asynchronously checks if the query is done by invoking the `index_done_callback` method
        on each storage instance in the `llm_response_cache`.

        This method gathers all the tasks and waits for them to complete.

        Returns:
            None
        """
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        """
        Deletes an entity by its name.

        Args:
            entity_name (str): The name of the entity to be deleted.

        Returns:
            The result of the asynchronous delete operation.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        """
        Asynchronously deletes an entity and its associated relationships from the database.

        Args:
            entity_name (str): The name of the entity to be deleted.

        Raises:
            Exception: If an error occurs while deleting the entity or its relationships.

        Logs:
            Info: When the entity and its relationships have been successfully deleted.
            Error: If an error occurs during the deletion process.
        """
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        """
        Asynchronously calls the `index_done_callback` method for each storage instance
        in the list of entities, relationships, and chunk entity relation graph.

        This method gathers the tasks for each non-None storage instance and waits for
        all of them to complete.

        Returns:
            None
        """
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
