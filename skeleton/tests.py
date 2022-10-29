from numpy import mean
from assignment_12 import *
import pytest

relation_1 = "../data/friends_toy.txt"
relation_2 = "../data/movie_ratings_toy.txt"

how_tuple_index = 0
lineage_tuple_index = 0

# Filter function
def relation_filter_left(scan_output_left):
    if(scan_output_left is not None):
        filter_output_left = [value for value in scan_output_left if(value.tuple[0] == 1)]
        return filter_output_left

def pull_recommendation(track_prov, propagate_prov):
    scan_operator_left = Scan(filepath=relation_1, outputs=[], relation_tag="L", track_prov=track_prov, propagate_prov = propagate_prov)
    scan_operator_right = Scan(filepath=relation_2, outputs=[], relation_tag="R", track_prov=track_prov, propagate_prov = propagate_prov)

    filter_operator_left = Select(inputs=[scan_operator_left], outputs=[], predicate=relation_filter_left, track_prov=track_prov, propagate_prov = propagate_prov)

    join_operator = Join(left_inputs=[filter_operator_left], right_inputs=[scan_operator_right], outputs=[], left_join_attribute=1, right_join_attribute=0, track_prov=track_prov, propagate_prov = propagate_prov)

    project_operator = Project(inputs=[join_operator], outputs=[], fields_to_keep=["MID", "Rating"], track_prov=track_prov, propagate_prov = propagate_prov)

    groupby_operator = GroupBy(inputs=[project_operator], outputs=[], agg_fun=mean, key=0, value=1, track_prov=track_prov, propagate_prov = propagate_prov)

    orderby_operator = OrderBy(inputs=[groupby_operator], outputs=[], comparator=sorted, ASC=False, track_prov=track_prov, propagate_prov = propagate_prov)

    limit_operator = TopK(inputs=[orderby_operator], outputs=[], k=1, track_prov=track_prov, propagate_prov = propagate_prov)

    project_operator = Project(inputs=[limit_operator], outputs=[], fields_to_keep=["MID"], aliasing=False, track_prov=track_prov, propagate_prov = propagate_prov)

    sink_operator = Sink(inputs=[project_operator], filepath="", track_prov=track_prov, propagate_prov = propagate_prov)
    sink_operator.get_next()

    temp = None

    if(track_prov):
        temp = sink_operator.write_lineage(lineage_tuple_index)

    if(propagate_prov):
        temp = sink_operator.write_how_provenance(how_tuple_index)
    
    return temp

def push_recommendation(track_prov, propagate_prov):
    sink_operator = Sink(inputs=[], filepath="", track_prov=track_prov, propagate_prov = propagate_prov)

    project_operator = Project(inputs=[], outputs=[sink_operator], fields_to_keep=["MID"], aliasing=False, track_prov=track_prov, propagate_prov = propagate_prov)

    limit_operator = TopK(inputs=[], outputs=[project_operator], k=1, track_prov=track_prov, propagate_prov = propagate_prov)

    orderby_operator = OrderBy(inputs=[], outputs=[limit_operator], comparator=sorted, ASC=False, track_prov=track_prov, propagate_prov = propagate_prov)

    groupby_operator = GroupBy(inputs=[], outputs=[orderby_operator], agg_fun=mean, key=0, value=1, track_prov=track_prov, propagate_prov = propagate_prov) # key and value are attribute numbers after projection

    project_operator = Project(inputs=[], outputs=[groupby_operator], fields_to_keep=["MID", "Rating"], track_prov=track_prov, propagate_prov = propagate_prov)

    join_operator = Join(left_inputs=[], right_inputs=[], outputs=[project_operator], left_join_attribute=1, right_join_attribute=0, track_prov=track_prov, propagate_prov = propagate_prov)

    filter_operator_left = Select(inputs=[], outputs=[join_operator], predicate=relation_filter_left, track_prov=track_prov, propagate_prov = propagate_prov) 

    scan_operator_left = Scan(filepath=relation_1, outputs=[filter_operator_left], relation_tag="L", track_prov=track_prov, propagate_prov = propagate_prov)
    scan_operator_right = Scan(filepath=relation_2, outputs=[join_operator], relation_tag="R", track_prov=track_prov, propagate_prov = propagate_prov)

    scan_operator_left.start()
    scan_operator_right.start()
    
    temp = None

    if(track_prov):
        temp = sink_operator.write_lineage(lineage_tuple_index)

    if(propagate_prov):
        temp = sink_operator.write_how_provenance(how_tuple_index)
    
    return temp




## -------------------------
## Pull-based
## -------------------------
def test_pull_lineage():
    correct_answer = [(1, 2), (2, 1, 4), (1, 3), (3, 1, 5), (1, 4), (4, 1, 4), (1, 5), (5, 1, 3)]
    temp = pull_recommendation(True, False)
    collected_output = []
    for item in temp:
        collected_output.append(item.tuple)
    assert(collected_output == correct_answer)

def test_pull_where_provenance():
    pass

def test_pull_how_provenance():
    correct_answer = "AVG( (f1*r3@4), (f2*r5@5), (f3*r7@4), (f4*r9@3) )"
    collected_output = pull_recommendation(False, True)
    assert(collected_output == correct_answer)
    

def test_pull_responsibility():
    pass

## -------------------------
## Push-based
## -------------------------
def test_push_lineage():
    correct_answer = [(1, 2), (2, 1, 4), (1, 3), (3, 1, 5), (1, 4), (4, 1, 4), (1, 5), (5, 1, 3)]
    temp = push_recommendation(True, False)
    collected_output = []
    for item in temp:
        collected_output.append(item.tuple)
    assert(collected_output == correct_answer)

def test_push_where_provenance():
    pass

def test_push_how_provenance():
    correct_answer = "AVG( (f1*r3@4), (f2*r5@5), (f3*r7@4), (f4*r9@3) )"
    collected_output = push_recommendation(False, True)
    assert(collected_output == correct_answer)

def test_push_responsibility():
    pass