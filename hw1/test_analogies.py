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
    # cosine similarity is the dot product of u and v when they are normalized to unit length
    # raise NotImplementedError("Problem 3b has not been completed yet!")
    # Hint: `cosine_sim` can be implemented in **at most 9 lines of code**. The performance of your code may suffer if it is substantially longer than this!
    
    # Calculate the L2 norm of each row in x and y
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)#(m, 1)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)#(n, 1)

    # Normalize x and y
    x_normalized = x/(x_norm)
    y_normalized = y/(y_norm)

    # dot product of x and y
    return np.dot(x_normalized, y_normalized.T)  # (m, n)
    


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
    # Hint: implemented in **at most 3 lines of code**.
    # Hint: You can make your code run faster if you don't try to return the closest words in order (though the difference in performance may not be noticeable since we are using a small number of embeddings).
    # raise NotImplementedError("Problem 3c has not been completed yet!")
    
    # Compute the cosine similarity between the vectors and the embeddings 
    # then get the indices of the top k closest words for each vector
    # closest_indices = np.argpartition(cosine_sim(vectors, embeddings.vectors), -k, axis=1)[:, -k:]
    closest_indices = np.argpartition(-cosine_sim(vectors, embeddings.vectors), k, axis=1)[:, :k]
    # Return the closest words
    return [[embeddings.words[i] for i in indices] for indices in closest_indices]

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
    # Hint: implemented in **at most 9 lines of code**
    # raise NotImplementedError("Problem 3d has not been completed yet!")
    
    analogy_test_results = {}
    # iterate over the relation types
    for relation, analogies in test_data.items():
        # iterate over the analogies
        # cur_correct = 0
        # for analogie in analogies:
        #     # get the vectors for the words in the analogy
        #     input_vector = embeddings[[analogie[2]]] - embeddings[[analogie[0]]] + embeddings[[analogie[1]]]
            
        #     # get the closest words for the vector
        #     closest_words = get_closest_words(embeddings, input_vector, k)
        
        #     # check if the last word in the analogy is in the closest words
        #     if analogie[3] in closest_words[0]:
        #         cur_correct += 1
        
        # # calculate the accuracy
        # analogy_test_results[relation] = cur_correct / len(analogies)

        # transform to array
        analogies_arr = np.array(analogies) # (N, 4)
        
        # get the vectors for the words in the analogys
        # analogies_arr[:,2] (N)
        input_vectors = embeddings[analogies_arr[:,2]] - embeddings[analogies_arr[:,0]] + embeddings[analogies_arr[:,1]] # (N, embedding_size)
        
        # get the closest words for the vectors
        closest_words = get_closest_words(embeddings, input_vectors, k) # (N, k)
        
        # check if the last word in the analogy is in the closest words
        analogy_test_results[relation] = sum([1 if analogies_arr[i,3] in closest_words[i] else 0 for i in range(len(analogies))])/len(analogies)
                
    return analogy_test_results
