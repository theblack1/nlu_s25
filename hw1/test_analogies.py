"""
Code for Problems 2 and 3 of HW 1.
"""
from typing import Dict, List, Tuple

import numpy as np

from embeddings import Embeddings


def cosine_sim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Problem 3b: Implement this function.

    Computes the cosine similarity between two matrices of row vectors.

    :param x: A 2D array of shape (m, embedding_size)
    :param y: A 2D array of shape (n, embedding_size)
    :return: An array of shape (m, n), where the entry in row i and
        column j is the cosine similarity between x[i] and y[j]
    """
    raise NotImplementedError("Problem 3b has not been completed yet!")


def get_closest_words(embeddings: Embeddings, vectors: np.ndarray,
                      k: int = 1) -> List[List[str]]:
    """
    Problem 3c: Implement this function.

    Finds the top k words whose embeddings are closest to a given vector
    in terms of cosine similarity.

    :param embeddings: A set of word embeddings
    :param vectors: A 2D array of shape (m, embedding_size)
    :param k: The number of closest words to find for each vector
    :return: A list of m lists of words, where the ith list contains the
        k words that are closest to vectors[i] in the embedding space,
        not necessarily in order
    """
    raise NotImplementedError("Problem 3c has not been completed yet!")


# This type alias represents the format that the testing data should be
# deserialized into. An analogy is a tuple of 4 strings, and an
# AnalogiesDataset is a dict that maps a relation type to the list of
# analogies under that relation type.
AnalogiesDataset = Dict[str, List[Tuple[str, str, str, str]]]


def load_analogies(filename: str) -> AnalogiesDataset:
    """
    Problem 2b: Implement this function.

    Loads testing data for 3CosAdd from a .txt file and deserializes it
    into the AnalogiesData format.

    :param filename: The name of the file containing the testing data
    :return: An AnalogiesDataset containing the data in the file. The
        format of the data is described in the problem set and in the
        docstring for the AnalogiesDataset type alias
    """
    
    # The keys of the `dict` must exactly match the names of the relation types listed in the `data/analogies.txt` data file. The keys should not include the initial `: ` in the names of the relation types, and they should not contain any leading or trailing whitespace.
    # Each analogy must be represented as a `tuple` or `list` of exactly 4 strings.
    # raise NotImplementedError("Problem 2b has not been completed yet!")
    
    # initialize
    analogies_dataset = {} 
    cur_relation = None
    
    with open(filename, 'r') as f:
        for line in f:
            line_striped = line.strip()
            
            # if line is empty, continue
            if not line_striped:
                continue
            
            # if line starts with ': ', it is a relation type
            if line_striped.startswith(':'):
                cur_relation = line_striped[2:]
                analogies_dataset[cur_relation] = []
            else:
                # the line is an analogy
                
                # Since glove dataset only contains lowercase words, we should lowercase the words
                line_striped = line_striped.lower()
                
                # Split the line into words
                line_striped_split = line_striped.split()
                # check if the line has 4 words
                if len(line_striped_split) != 4:
                    raise ValueError(f"Invalid analogy: {line_striped}")
                analogy = tuple(line_striped_split)
                analogies_dataset[cur_relation].append(analogy)
    
    return analogies_dataset


def run_analogy_test(embeddings: Embeddings, test_data: AnalogiesDataset,
                     k: int = 1) -> Dict[str, float]:
    """
    Problem 3d: Implement this function.

    Runs the 3CosAdd test for a set of embeddings and analogy questions.

    :param embeddings: The embeddings to be evaluated using 3CosAdd
    :param test_data: The set of analogies with which to compute analogy
        question accuracy
    :param k: The "lenience" with which accuracy will be computed. An
        analogy question will be considered to have been answered
        correctly as long as the target word is within the top k closest
        words to the target vector, even if it is not the closest word
    :return: The results of the 3CosAdd test, in the format of a dict
        that maps each relation type to the analogy question accuracy
        attained by embeddings on analogies from that relation type
    """
    raise NotImplementedError("Problem 3d has not been completed yet!")
