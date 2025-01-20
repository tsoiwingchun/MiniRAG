import asyncio
import html
import os
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB
import copy
from .utils import (
    logger,
    load_json,
    write_json,
    compute_mdhash_id,
    merge_tuples,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    """
    A key-value storage class that uses JSON files for persistence.

    Attributes:
        global_config (dict): A dictionary containing global configuration.
        namespace (str): The namespace for the key-value store.
        _file_name (str): The file path for the JSON file storing the key-value data.
        _data (dict): The in-memory dictionary storing the key-value data.

    Methods:
        __post_init__():
            Initializes the storage by loading data from the JSON file.
        
        all_keys() -> list[str]:
            Asynchronously returns a list of all keys in the storage.
        
        index_done_callback():
            Asynchronously writes the current in-memory data to the JSON file.
        
        get_by_id(id):
            Asynchronously retrieves the value associated with the given key.
        
        get_by_ids(ids, fields=None):
            Asynchronously retrieves values for the given list of keys. If fields are specified, only those fields are returned.
        
        filter_keys(data: list[str]) -> set[str]:
            Asynchronously filters out keys that already exist in the storage from the given list.
        
        upsert(data: dict[str, dict]):
            Asynchronously inserts or updates the given key-value pairs in the storage. Returns the data that was inserted.
        
        drop():
            Asynchronously clears all data in the storage.
    """
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.#2

    def __post_init__(self):
        """
        Post-initialization method for setting up the storage client and configuration.

        This method is automatically called after the object is initialized. It performs
        the following tasks:
        - Constructs the file path for the client storage file using the working directory
          and namespace.
        - Retrieves the maximum batch size for embeddings from the global configuration.
        - Initializes the NanoVectorDB client with the specified embedding dimension and
          storage file path.
        - Sets the cosine similarity threshold from the global configuration, if provided.

        Attributes:
            _client_file_name (str): The file path for the client storage file.
            _max_batch_size (int): The maximum number of embeddings to process in a batch.
            _client (NanoVectorDB): The NanoVectorDB client instance for storage operations.
            cosine_better_than_threshold (float): The threshold for cosine similarity.
        """
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        """
        Upserts the given data into the vector database.

        Args:
            data (dict[str, dict]): A dictionary where the key is a unique identifier and the value is another dictionary containing the data to be upserted.

        Returns:
            list: A list of results from the upsert operation.

        Raises:
            ValueError: If the data dictionary is empty.

        Logs:
            Logs the number of vectors being inserted.
            Logs a warning if the data dictionary is empty.
        """
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        """
        Asynchronously queries the storage with the given query string and returns the top-k results.

        Args:
            query (str): The query string to search for.
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            list: A list of dictionaries containing the search results, each with an 'id' and 'distance' key.
        """
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )

        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    @property
    def client_storage(self):
        """
        Retrieve the storage attribute from the _client object.

        This method accesses the private attribute __storage of the _NanoVectorDB
        instance stored in the _client attribute of the current object.

        Returns:
            The storage attribute of the _NanoVectorDB instance.
        """
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete_entity(self, entity_name: str):
        """
        Asynchronously deletes an entity by its name.

        Args:
            entity_name (str): The name of the entity to be deleted.

        Raises:
            Exception: If an error occurs during the deletion process.

        Logs:
            - Info: If the entity is successfully deleted or if no entity is found with the given name.
            - Error: If an error occurs while attempting to delete the entity.
        """
        try:
            entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

            if self._client.get(entity_id):
                self._client.delete(entity_id)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        """
        Asynchronously deletes all relations associated with a given entity.

        Args:
            entity_name (str): The name of the entity whose relations are to be deleted.

        Raises:
            Exception: If an error occurs while deleting the relations.

        Logs:
            Info: When all relations related to the entity are successfully deleted or if no relations are found.
            Error: If an error occurs while deleting the relations.
        """
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(
                    f"All relations related to entity {entity_name} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        """
        Load a NetworkX graph from a GraphML file.

        Parameters:
        file_name (str): The path to the GraphML file.

        Returns:
        nx.Graph: The loaded NetworkX graph if the file exists, otherwise None.
        """
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        """
        Writes a NetworkX graph to a file in GraphML format.

        Parameters:
        graph (nx.Graph): The NetworkX graph to be written to the file.
        file_name (str): The name of the file where the graph will be saved.

        Returns:
        None
        """
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """
         This function takes a NetworkX graph as input, finds its largest connected component, and returns it with nodes
            and edges sorted in a stable manner. The node labels are converted to uppercase and stripped of leading/trailing
            whitespace.

            Args:
                graph (nx.Graph): The input graph from which the largest connected component is to be extracted.

            Returns:
                nx.Graph: The largest connected component of the input graph with nodes and edges sorted stably.
            Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """
        Stabilize the given graph to ensure consistent ordering of nodes and edges.

        This function takes an undirected or directed graph and returns a new graph
        with nodes and edges sorted in a consistent manner. This ensures that the
        graph will always be read the same way, regardless of the initial order of
        nodes and edges.

        Args:
            graph (nx.Graph): The input graph to be stabilized.

        Returns:
            nx.Graph: A new graph with nodes and edges sorted in a consistent order.

        References:
            https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        """
        Post-initialization method for setting up the graph storage.

        This method constructs the file path for the GraphML XML file based on the
        working directory and namespace provided in the global configuration. It then
        attempts to load a pre-existing graph from this file. If a graph is successfully
        loaded, it logs the number of nodes and edges in the graph. If no graph is loaded,
        it initializes an empty NetworkX graph. Additionally, it sets up a dictionary of
        node embedding algorithms.

        Attributes:
            _graphml_xml_file (str): The file path for the GraphML XML file.
            _graph (networkx.Graph): The loaded or newly initialized NetworkX graph.
            _node_embed_algorithms (dict): A dictionary mapping algorithm names to their
                                           corresponding embedding functions.
        """
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        """
        Asynchronous callback function to handle the completion of an indexing operation.

        This function writes the current state of the NetworkX graph to a GraphML XML file.

        Returns:
            None
        """
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the given ID exists in the graph.

        Args:
            node_id (str): The ID of the node to check.

        Returns:
            bool: True if the node exists in the graph, False otherwise.
        """
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if there is an edge between two nodes in the graph.

        Args:
            source_node_id (str): The ID of the source node.
            target_node_id (str): The ID of the target node.

        Returns:
            bool: True if there is an edge from the source node to the target node, False otherwise.
        """
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """
        Retrieve a node from the graph by its ID.

        Args:
            node_id (str): The unique identifier of the node to retrieve.

        Returns:
            Union[dict, None]: The node data as a dictionary if found, otherwise None.
        """
        return self._graph.nodes.get(node_id)
    
    async def get_types(self) -> list:
        """
        Asynchronously retrieves the types of entities and their corresponding names from the graph.

        Returns:
            tuple: A tuple containing:
                - list: A list of unique entity types.
                - dict: A dictionary where the keys are entity types and the values are lists of entity names.
        """
        all_entity_type = []
        all_type_w_name = {}
        for n in self._graph.nodes(data=True):
            key = n[1]['entity_type'].strip('\"')
            all_entity_type.append(key)
            if key not in all_type_w_name:
                all_type_w_name[key] = []
                all_type_w_name[key].append(n[0].strip('\"'))
            else:
                if len(all_type_w_name[key])<=1:
                    all_type_w_name[key].append(n[0].strip('\"'))

        return list(set(all_entity_type)),all_type_w_name
    


    async def get_node_from_types(self,type_list)  -> Union[dict, None]:
        """
        Asynchronously retrieves nodes from the graph that match the specified types.

        Args:
            type_list (list): A list of node types to filter by.

        Returns:
            Union[dict, None]: A list of dictionaries containing node data for nodes
            that match the specified types, or None if no matching nodes are found.
        """
        node_list = []
        for name, arrt in self._graph.nodes(data = True):
            node_type = arrt.get('entity_type').strip('\"')
            if node_type in type_list:
                node_list.append(name)
        node_datas = await asyncio.gather(
            *[self.get_node(name) for name in node_list]
        )
        node_datas = [
            {**n, "entity_name": k}
            for k, n in zip(node_list, node_datas)
            if n is not None
        ]
        return node_datas#,node_dict
    

    async def get_neighbors_within_k_hops(self,source_node_id: str, k):
        """
        Asynchronously retrieves the neighbors of a given node within `k` hops in the graph.

        Args:
            source_node_id (str): The ID of the source node from which to start the search.
            k (int): The number of hops to search for neighbors.

        Returns:
            list: A list of edges representing the neighbors within `k` hops from the source node.
                  Each edge is represented as a tuple of node IDs.

        Raises:
            None

        Notes:
            - If the source node does not exist in the graph, an empty list is returned.
            - The function uses a breadth-first search approach to find neighbors within `k` hops.
        """
        count = 0
        if await self.has_node(source_node_id):
            source_edge = list(self._graph.edges(source_node_id))
        else:
            print("NO THIS ID:",source_node_id)
            return []
        count = count+1
        while count<k:
            count = count+1
            sc_edge = copy.deepcopy(source_edge)
            source_edge =[]
            for pair in sc_edge:
                append_edge = list(self._graph.edges(pair[-1]))
                for tuples in merge_tuples([pair],append_edge):
                    source_edge.append(tuples)
        return source_edge
    async def node_degree(self, node_id: str) -> int:
        """
        Asynchronously retrieves the degree of a specified node in the graph.

        The degree of a node is the number of edges connected to it.

        Args:
            node_id (str): The identifier of the node whose degree is to be retrieved.

        Returns:
            int: The degree of the specified node.
        """
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """
        Calculate the sum of the degrees of two nodes in the graph.

        Args:
            src_id (str): The source node identifier.
            tgt_id (str): The target node identifier.

        Returns:
            int: The sum of the degrees of the source and target nodes.
        """
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        """
        Retrieve the edge data between two nodes in the graph.

        Args:
            source_node_id (str): The ID of the source node.
            target_node_id (str): The ID of the target node.

        Returns:
            Union[dict, None]: The edge data as a dictionary if the edge exists, 
                               otherwise None.
        """
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        """
        Asynchronously retrieves the edges connected to a given source node in the graph.

        Args:
            source_node_id (str): The ID of the source node whose edges are to be retrieved.

        Returns:
            list: A list of edges connected to the source node if the node exists in the graph.
            None: If the source node does not exist in the graph.
        """
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """
        Asynchronously upserts a node into the graph.

        If the node with the given node_id already exists, it updates the node with the provided node_data.
        If the node does not exist, it creates a new node with the given node_id and node_data.

        Args:
            node_id (str): The unique identifier for the node.
            node_data (dict[str, str]): A dictionary containing the node's attributes and their values.

        Returns:
            None
        """
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        """
        Asynchronously upserts an edge between two nodes in the graph.

        If an edge already exists between the specified source and target nodes,
        it will be updated with the provided edge data. If no such edge exists,
        a new edge will be created.

        Args:
            source_node_id (str): The ID of the source node.
            target_node_id (str): The ID of the target node.
            edge_data (dict[str, str]): A dictionary containing the edge data to be added or updated.

        Returns:
            None
        """
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str):
        """
        Delete a node from the graph based on the specified node_id.

        :param node_id: The node_id to delete
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        """
        Embed nodes using the specified algorithm.

        Args:
            algorithm (str): The name of the node embedding algorithm to use.

        Returns:
            tuple[np.ndarray, list[str]]: A tuple containing the embedded nodes as a numpy array and a list of corresponding node identifiers.

        Raises:
            ValueError: If the specified algorithm is not supported.
        """
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        """
        Asynchronously generates node embeddings using the Node2Vec algorithm.

        This method uses the graspologic library to compute embeddings for the nodes
        in the graph stored in `self._graph`. The parameters for the Node2Vec algorithm
        are retrieved from `self.global_config["node2vec_params"]`.

        Returns:
            tuple: A tuple containing:
                - embeddings (numpy.ndarray): The computed node embeddings.
                - nodes_ids (list): A list of node IDs corresponding to the embeddings.
        """
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
