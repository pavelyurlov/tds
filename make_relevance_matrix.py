import re

import numpy as np

from east.asts import base

import json


def clear_text(text, lowerise=True):

    pat = re.compile(r'[^A-Za-z0-9 \-\n\r.,;!?А-Яа-я]+')
    cleared_text = re.sub(pat, ' ', text)

    if lowerise:
        cleared_text = cleared_text.lower()

    tokens = cleared_text.split()
    return tokens


def make_substrings(tokens, k=4):

    for i in range(max(len(tokens) - k + 1, 1)):
        yield ' '.join(tokens[i:i + k])


def get_relevance_matrix(texts, strings):

    matrix = np.empty((0, len(strings)), float)
    prepared_text_tokens = [clear_text(t) for t in texts]

    prepared_string_tokens = [clear_text(s) for s in strings]
    prepared_strings = [' '.join(t) for t in prepared_string_tokens]

    for text_tokens in prepared_text_tokens:
        ast = base.AST.get_ast(list(make_substrings(text_tokens)))
        row = np.array([ast.score(s) for s in prepared_strings])
        matrix = np.append(matrix, [row], axis=0)

    return matrix


def save_matrix(matrix):
    np.savetxt("relevance_matrix.txt", matrix)


if __name__ == "__main__":

    with open("ml_abstracts.txt", "r") as f:
        abstracts = json.loads(f.read())

    with open("taxonomy_leaves.txt") as f:
        strings = [l.strip() for l in f.readlines()]

    relevance_matrix = get_relevance_matrix(abstracts, strings)
    save_matrix(relevance_matrix)

