import asyncio
import html
import io
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List
import xml.etree.ElementTree as ET
import copy
import numpy as np
import tiktoken

ENCODER = None

logger = logging.getLogger("minirag")


def set_logger(log_file: str):  # to configure the logging settings for the application. It sets up a logger to write log messages to a specified file, ensuring that detailed information about the application's execution is recorded
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)


@dataclass
class EmbeddingFunc: #to encapsulate the functionality and parameters required to generate embeddings asynchronously
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        return None


def convert_response_to_json(response: str) -> dict:   #converts a string response into a JSON object 
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def compute_args_hash(*args):  #computes a hash value for its input arguments.
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content, prefix: str = ""):
    """
    Computes an MD5 hash of the given content and returns it as a string, optionally prefixed.

    Args:
        content (str): The content to hash.
        prefix (str, optional): A string to prefix the hash with. Defaults to an empty string.

    Returns:
        str: The MD5 hash of the content, prefixed with the given prefix.
    """
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """
    Decorator to limit the number of concurrent asynchronous function calls.

    Args:
        max_size (int): The maximum number of concurrent calls allowed.
        waitting_time (float, optional): The time to wait before retrying if the limit is reached. Defaults to 0.0001 seconds.

    Returns:
        Callable: A decorator that limits the number of concurrent calls to the decorated async function.
    """
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """
    Wrap a function with attributes.

    This decorator takes keyword arguments and wraps a function with these attributes
    by creating an instance of `EmbeddingFunc` with the provided attributes and the
    original function.

    Args:
        **kwargs: Arbitrary keyword arguments to be passed as attributes to the `EmbeddingFunc`.

    Returns:
        function: A decorator that wraps the original function with the specified attributes.
    """
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    """
    Load a JSON file and return its contents as a Python object.

    Args:
        file_name (str): The path to the JSON file to be loaded.

    Returns:
        dict or list: The contents of the JSON file as a Python dictionary or list.
        None: If the file does not exist.
    """
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    """
    Write a JSON object to a file.

    Args:
        json_obj (dict): The JSON object to write to the file.
        file_name (str): The name of the file to write the JSON object to.

    Returns:
        None
    """
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    """
    Encodes a given string using the tiktoken library based on the specified model.

    Args:
        content (str): The string content to be encoded.
        model_name (str, optional): The name of the model to use for encoding. Defaults to "gpt-4o".

    Returns:
        list: A list of tokens representing the encoded string.
    """
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    """
    Decodes a list of tokens into a string using the specified model's encoding.

    Args:
        tokens (list[int]): A list of integer tokens to be decoded.
        model_name (str, optional): The name of the model to use for decoding. Defaults to "gpt-4o".

    Returns:
        str: The decoded string content.
    """
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def pack_user_ass_to_openai_messages(*args: str):
    """
    Packs user and assistant messages into a format suitable for OpenAI API.

    Args:
        *args (str): A variable number of string arguments representing the messages.

    Returns:
        list: A list of dictionaries, each containing a "role" (either "user" or "assistant") 
              and "content" (the message content). The roles alternate starting with "user".
    """
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """
    Split a string by multiple markers.

    Args:
        content (str): The string to be split.
        markers (list[str]): A list of marker strings to split the content by.

    Returns:
        list[str]: A list of substrings obtained by splitting the content by the markers.
    """
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """
    Clean an input string by removing HTML escapes, control characters, and other unwanted characters.

    Args:
        input (Any): The input to be cleaned. If the input is not a string, it will be returned as is.

    Returns:
        str: The cleaned string with HTML escapes and control characters removed.
    """
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value):
    """
    Check if the given value is a valid float using a regular expression.

    Args:
        value (str): The value to be checked.

    Returns:
        bool: True if the value is a valid float, False otherwise.
    """
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """
    Truncate a list of data by token size.

    This function iterates through a list of data, applying a key function to each element
    to obtain a string representation. It then encodes the string representation to tokens
    and accumulates the token count. Once the accumulated token count exceeds the specified
    maximum token size, the function returns a truncated list up to the current element.

    Args:
        list_data (list): The list of data to be truncated.
        key (callable): A function that takes an element from the list and returns a string.
        max_token_size (int): The maximum allowed token size for the truncated list.

    Returns:
        list: A truncated list of data where the total token size does not exceed max_token_size.
    """
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_csv(data: List[List[str]]) -> str:
    """
    Converts a list of lists into a CSV formatted string.

    Args:
        data (List[List[str]]): A list of lists where each inner list represents a row of data.

    Returns:
        str: A string in CSV format representing the input data.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> List[List[str]]:
    """
    Convert a CSV formatted string into a list of lists.

    Args:
        csv_string (str): A string containing CSV formatted data.

    Returns:
        List[List[str]]: A list of lists where each inner list represents a row of the CSV data.
    """
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]


def save_data_to_file(data, file_name):
    """
    Save the given data to a file in JSON format.

    Args:
        data (dict): The data to be saved.
        file_name (str): The name of the file where the data will be saved.

    Raises:
        IOError: If the file cannot be opened or written to.
    """
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file):
    """
    Convert an XML file to a JSON-like dictionary.

    Args:
        xml_file (str): Path to the XML file to be converted.

    Returns:
        dict: A dictionary containing nodes and edges extracted from the XML file.
              The dictionary has the following structure:
              {
                  "nodes": [
                      {
                          "id": str,
                          "entity_type": str,
                          "description": str,
                          "source_id": str
                      },
                      ...
                  ],
                  "edges": [
                      {
                          "source": str,
                          "target": str,
                          "weight": float,
                          "description": str,
                          "keywords": str,
                          "source_id": str
                      },
                      ...
                  ]
              Returns None if an error occurs during parsing.

    Raises:
        ET.ParseError: If there is an error parsing the XML file.
        Exception: For any other exceptions that occur during processing.

    Example:
        data = xml_to_json("path/to/your/file.xml")
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_combine_contexts(hl, ll):
    """
    Processes and combines two CSV string contexts into a single formatted string.

    Args:
        hl (str): The first CSV string context.
        ll (str): The second CSV string context.

    Returns:
        str: A combined and formatted string of the two CSV contexts. If both contexts are empty, returns an empty string.

    The function performs the following steps:
    1. Converts the CSV strings to lists.
    2. Extracts the header from the first non-empty list.
    3. Removes the header from both lists.
    4. Combines the remaining items from both lists, excluding the first column.
    5. Filters out any empty items and duplicates.
    6. Formats the combined items with a header and index.
    """
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources_set = set(filter(None, list_hl + list_ll))

    combined_sources = [",\t".join(header)]

    for i, item in enumerate(combined_sources_set, start=1):
        combined_sources.append(f"{i},\t{item}")

    combined_sources = "\n".join(combined_sources)

    return combined_sources




def is_continuous_subsequence(subseq, seq):
    """
    Check if `subseq` is a continuous subsequence of `seq`.

    A continuous subsequence means that the elements of `subseq` appear in the same order and are contiguous in `seq`.

    Args:
        subseq (list): The subsequence to check.
        seq (list): The sequence in which to check for the subsequence.

    Returns:
        bool: True if `subseq` is a continuous subsequence of `seq`, False otherwise.
    """
    def find_all_indexes(tup, value):
        indexes = []
        start = 0
        while True:
            try:
                index = tup.index(value, start)
                indexes.append(index)
                start = index + 1
            except ValueError:
                break
        return indexes

    index_list = find_all_indexes(seq,subseq[0])
    for idx in index_list:
        if idx!=len(seq)-1:
            if seq[idx+1]==subseq[-1]:
                return True
    return False



def merge_tuples(list1, list2):
    """
    Merges tuples from two lists based on specific conditions.
    This function takes two lists of tuples and merges tuples from the first list
    with tuples from the second list if certain conditions are met. Specifically,
    it checks if the last element of a tuple in the first list matches the first
    element of any tuple in the second list. If a match is found and the merged
    tuple does not form a continuous subsequence, the tuples are merged.
    Args:
        list1 (list of tuples): The first list of tuples to be merged.
        list2 (list of tuples): The second list of tuples to be merged.
    Returns:
        list of tuples: A list of merged tuples based on the specified conditions.
    """
    result = []
    for tup in list1:
    
        last_element = tup[-1]
        if last_element in tup[:-1]:
            result.append(tup)
        else:

            matching_tuples = [t for t in list2 if t[0] == last_element]
   
            already_match_flag = 0
            for match in matching_tuples:

                matchh = (match[1],match[0])
                if is_continuous_subsequence(match, tup) or is_continuous_subsequence(matchh, tup):
                    continue  
                
                already_match_flag = 1
                merged_tuple = tup + match[1:]

                result.append(merged_tuple)
            
            if not already_match_flag:
                result.append(tup)
    return result


def count_elements_in_tuple(tuple_elements, list_elements):
    """
    Count the number of elements in `tuple_elements` that are also present in `list_elements`.
    This function first sorts both the input tuple and list, then iterates through the sorted tuple,
    counting how many of its elements are present in the sorted list.
    Args:
        tuple_elements (tuple): A tuple of elements to be counted.
        list_elements (list): A list of elements to be checked against.
    Returns:
        int: The count of elements from `tuple_elements` that are found in `list_elements`.
    """
    sorted_list = sorted(list_elements)
    tuple_elements = sorted(tuple_elements)
    count = 0
    list_index = 0
    
    for elem in tuple_elements:
        while list_index < len(sorted_list) and sorted_list[list_index] < elem:
            list_index += 1
        if list_index < len(sorted_list) and sorted_list[list_index] == elem:
            count += 1
            list_index += 1  
    return count


def cal_path_score_list(candidate_reasoning_path, maybe_answer_list):
    """
    Calculate the score list for candidate reasoning paths.

    Args:
        candidate_reasoning_path (dict): A dictionary where keys are identifiers and values are dictionaries 
                                         containing 'Score' and 'Path'. 'Score' is a numerical value and 'Path' 
                                         is a list of tuples representing reasoning paths.
        maybe_answer_list (list): A list of possible answers to be used for scoring the reasoning paths.

    Returns:
        dict: A dictionary where keys are the same identifiers from the input dictionary and values are 
              dictionaries containing 'Score' and 'Path'. 'Score' is the original score and 'Path' is a 
              dictionary where keys are the original paths and values are lists containing the count of 
              elements in the path that are present in the maybe_answer_list.
    """
    scored_reasoning_path = {}
    for k,v in candidate_reasoning_path.items():
        score = v['Score']
        paths = v['Path']
        scores = {}
        for p in paths:
            scores[p] = [count_elements_in_tuple(p, maybe_answer_list)]
        scored_reasoning_path[k] = {'Score': score, 'Path': scores}
    return scored_reasoning_path




def edge_vote_path(path_dict,edge_list):
    def edge_vote_path(path_dict, edge_list):
        """
        Processes a dictionary of paths and an edge list to count occurrences of edge pairs in the paths.
        Args:
            path_dict (dict): A dictionary where keys are identifiers and values are dictionaries containing a 'Path' key.
                              The 'Path' key maps to another dictionary where keys are sequences and values are lists.
            edge_list (list): A list of dictionaries, each containing 'src_id' and 'tgt_id' keys representing edges.
        Returns:
            tuple: A tuple containing:
                - return_dict (dict): A deep copy of the input path_dict with counts appended to the path sequences.
                - pairs_append (dict): A dictionary where keys are path sequences and values are lists of edge pairs 
                                       that are continuous subsequences of the path sequences.
        """
    return_dict = copy.deepcopy(path_dict)
    EDGELIST = []
    pairs_append = {}
    for i in edge_list:
        EDGELIST.append((i['src_id'],i['tgt_id']))
    for i in return_dict.items():
        for j in i[1]['Path'].items():
            if j[1]:
                count = 0
                
                for pairs in EDGELIST:
                    
                    if is_continuous_subsequence(pairs,j[0]):
                        count = count+1
                        if j[0] not in pairs_append:
                            pairs_append[j[0]] = [pairs]
                        else:
                            pairs_append[j[0]].append(pairs)

                #score
                j[1].append(count)
    return return_dict,pairs_append



from nltk.metrics import edit_distance
from rouge import Rouge

def calculate_similarity(sentences, target, method='levenshtein', n=1, k=1):
    """
    Calculate the similarity between a list of sentences and a target sentence using different methods.
    Args:
        sentences (list of str): List of sentences to compare against the target.
        target (str): The target sentence to compare.
        method (str, optional): The method to use for calculating similarity. 
                                Options are 'levenshtein', 'jaccard', or 'rouge'. Default is 'levenshtein'.
        n (int, optional): The n-gram size for the ROUGE method. Default is 1.
        k (int, optional): The number of top similar sentences to return. Default is 1.
    Returns:
        list of int: Indices of the top k most similar sentences in the input list.
    Raises:
        ValueError: If an unsupported method is provided.
    """
    target_tokens = target.lower().split()
    similarities_with_index = []
    
    if method == 'jaccard':
        for i, sentence in enumerate(sentences):
            sentence_tokens = sentence.lower().split()
            intersection = set(sentence_tokens).intersection(set(target_tokens))
            union = set(sentence_tokens).union(set(target_tokens))
            jaccard_score = len(intersection) / len(union) if union else 0
            similarities_with_index.append((i, jaccard_score))
    
    elif method == 'levenshtein':
        for i, sentence in enumerate(sentences):
            distance = edit_distance(target_tokens, sentence.lower().split())
            similarities_with_index.append((i, 1 - (distance / max(len(target_tokens), len(sentence.split())))))
    
    elif method == 'rouge':
        rouge = Rouge()
        for i, sentence in enumerate(sentences):
            scores = rouge.get_scores(sentence, target)
            rouge_score = scores[0].get(f'rouge-{n}', {}).get('f', 0)
            similarities_with_index.append((i, rouge_score))
    
    else:
        raise ValueError("Unsupported method. Choose 'jaccard', 'levenshtein', or 'rouge'.")
    
    similarities_with_index.sort(key=lambda x: x[1], reverse=True)
    return [index for index, score in similarities_with_index[:k]]