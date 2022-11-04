from numpy import mean
from assignment_12 import *
import pytest

relation_1 = "../data/friends_toy.txt"
relation_2 = "../data/movie_ratings_toy.txt"

how_tuple_index = 0
lineage_tuple_index = 0
where_row_index = 0
where_attribute_index = 0

# Filter function
def relation_filter_left(scan_output_left):
    if(scan_output_left is not None):
        filter_output_left = [value for value in scan_output_left if(value.tuple[0] == 1)]
        return filter_output_left

def relation_filter_right(scan_output_right):
    if(scan_output_right is not None):
        filter_output_right = [value for value in scan_output_right if(value.tuple[1] == 1)]
        return filter_output_right

## -------------------------
## Pull-based
## -------------------------
def test_pull_lineage():
    correct_answer = [(1, 2), (2, 1, 4), (1, 3), (3, 1, 5), (1, 4), (4, 1, 4), (1, 5), (5, 1, 3)]
    temp = pull_recommendation(relation_1, relation_2, relation_filter_left, "", True, False, 0, -1, -1)
    collected_output = []
    for item in temp:
        collected_output.append(item.tuple)
    assert(collected_output == correct_answer)

def test_pull_where_provenance():
    correct_answer = [('movie_ratings_toy.txt', 3, (2, 1, 4), 4), ('movie_ratings_toy.txt', 5, (3, 1, 5), 5), ('movie_ratings_toy.txt', 7, (4, 1, 4), 4), ('movie_ratings_toy.txt', 9, (5, 1, 3), 3)]
    collected_output = pull_rating(relation_1, relation_2, relation_filter_left, relation_filter_right, "", True, where_row_index, where_attribute_index)
    assert(collected_output == correct_answer)

def test_pull_how_provenance():
    correct_answer = "AVG( (f1*r3@4), (f2*r5@5), (f3*r7@4), (f4*r9@3) )"
    collected_output = pull_recommendation(relation_1, relation_2, relation_filter_left, "", False, True, -1, 0, -1)
    assert(collected_output == correct_answer)
    

def test_pull_responsibility():
    correct_answer = [((1, 5), 0.5), ((1, 4), 0.5), ((3, 1, 5), 0.5), ((4, 2, 2), 0.5), ((5, 2, 2), 0.5)]
    collected_output = pull_recommendation(relation_1, relation_2, relation_filter_left, "", False, True, -1, 0, 0)
    assert(collected_output == correct_answer)

## -------------------------
## Push-based
## -------------------------
def test_push_lineage():
    correct_answer = [(1, 2), (2, 1, 4), (1, 3), (3, 1, 5), (1, 4), (4, 1, 4), (1, 5), (5, 1, 3)]
    temp = push_recommendation(relation_1, relation_2, relation_filter_left, "", True, False, 0, -1, -1)
    collected_output = []
    for item in temp:
        collected_output.append(item.tuple)
    assert(collected_output == correct_answer)

def test_push_where_provenance():
    correct_answer = [('movie_ratings_toy.txt', 3, (2, 1, 4), 4), ('movie_ratings_toy.txt', 5, (3, 1, 5), 5), ('movie_ratings_toy.txt', 7, (4, 1, 4), 4), ('movie_ratings_toy.txt', 9, (5, 1, 3), 3)]
    collected_output = push_rating(relation_1, relation_2, relation_filter_left, relation_filter_right, "", True, where_row_index, where_attribute_index)
    assert(collected_output == correct_answer)

def test_push_how_provenance():
    correct_answer = "AVG( (f1*r3@4), (f2*r5@5), (f3*r7@4), (f4*r9@3) )"
    collected_output = push_recommendation(relation_1, relation_2, relation_filter_left, "", False, True, -1, 0, -1)
    assert(collected_output == correct_answer)

def test_push_responsibility():
    correct_answer = [((1, 5), 0.5), ((1, 4), 0.5), ((3, 1, 5), 0.5), ((4, 2, 2), 0.5), ((5, 2, 2), 0.5)]
    collected_output = push_recommendation(relation_1, relation_2, relation_filter_left, "", False, True, -1, 0, 0)
    assert(collected_output == correct_answer)