from assignment_12 import Scan, Sink, Select, Join, Project
import pytest

friends_relation = "../data/friends_toy.txt"
movies_relation = "../data/movie_ratings_toy.txt"

# Filter function
def relation_filter(scan_output):
    if(scan_output is not None):
        filter_output_right = [value for value in scan_output if(value.tuple[1] == 2)]
        return filter_output_right



## -------------------------
## Pull-based
## -------------------------
def test_pull_scan_operator():
    scan_operator = Scan(filepath = friends_relation, outputs=[])
    correct_result = [(1,2), (1,3), (1,4), (1,5)]

    scan_operator_output = scan_operator.get_next()
    
    result_to_check = []
    for item in scan_operator_output[1]:
        result_to_check.append(item.tuple)
    
    assert(correct_result == result_to_check)

def test_pull_select_operator():
    scan_operator = Scan(filepath = friends_relation, outputs=[])
    select_operator = Select(inputs=[scan_operator], outputs=[], predicate=relation_filter)
    correct_result = [(1,2)]

    select_operator_output = select_operator.get_next()
    
    result_to_check = []
    for item in select_operator_output[1]:
        result_to_check.append(item.tuple)
    
    assert(correct_result == result_to_check)

def test_pull_join_operator():
    scan_operator_1 = Scan(filepath = friends_relation, outputs=[])
    scan_operator_2 = Scan(filepath = movies_relation, outputs=[])

    join_operator = Join(left_inputs=[scan_operator_1], right_inputs=[scan_operator_2], outputs=[], left_join_attribute=1, right_join_attribute=0)

    correct_result = [((1,2),(2,1,4),), ((1,2),(2,2,5),),
                      ((1,3),(3,1,5),), ((1,3),(3,2,5),),
                      ((1,4),(4,1,4),), ((1,4),(4,2,2),),
                      ((1,5),(5,1,3),), ((1,5),(5,2,2),)
                      ]

    join_operator_output = join_operator.get_next()
    
    result_to_check = []
    for item in join_operator_output[1]:
        result_to_check.append(item.tuple)
    
    assert(correct_result == result_to_check)

def test_pull_project_operator():
    scan_operator = Scan(filepath = friends_relation, outputs=[])
    project_operator = Project(inputs=[scan_operator], outputs=[], fields_to_keep=["UID2"])
    correct_result = [(2,), (3,), (4,), (5,)]

    project_operator_output = project_operator.get_next()
    
    result_to_check = []
    for item in project_operator_output[1]:
        result_to_check.append(item.tuple)
    
    assert(correct_result == result_to_check)


## -------------------------
## Push-based
## -------------------------
def test_push_scan_operator():
    sink_operator = Sink(inputs=[])
    scan_operator = Scan(filepath = friends_relation, outputs=[sink_operator])
    correct_result = [(1,2), (1,3), (1,4), (1,5)]

    scan_operator.start()
    sink_output = sink_operator.output_data

    result_to_check = []
    for item in sink_output:
        result_to_check.append(item.tuple)
    
    assert(correct_result == result_to_check)

def test_push_select_operator():
    sink_operator = Sink(inputs=[])
    select_operator = Select(outputs=[sink_operator], inputs=[], predicate=relation_filter)
    scan_operator = Scan(filepath = friends_relation, outputs=[select_operator])
    
    correct_result = [(1,2)]

    scan_operator.start()
    sink_output = sink_operator.output_data

    result_to_check = []
    for item in sink_output:
        result_to_check.append(item.tuple)
    
    assert(correct_result == result_to_check)

def test_push_join_operator():
    sink_operator = Sink(inputs=[])
    join_operator = Join(left_inputs=[], right_inputs=[], outputs=[sink_operator], left_join_attribute=1, right_join_attribute=0)

    scan_operator_1 = Scan(filepath=friends_relation, outputs=[join_operator], relation_tag="L")
    scan_operator_2 = Scan(filepath=movies_relation, outputs=[join_operator], relation_tag="R")

    scan_operator_1.start()
    scan_operator_2.start()
    
    correct_result = [((1,2),(2,1,4),), ((1,2),(2,2,5),),
                      ((1,3),(3,1,5),), ((1,3),(3,2,5),),
                      ((1,4),(4,1,4),), ((1,4),(4,2,2),),
                      ((1,5),(5,1,3),), ((1,5),(5,2,2),)
                      ]
    sink_output = sink_operator.output_data
    
    result_to_check = []
    for item in sink_output:
        result_to_check.append(item.tuple)
    
    assert(correct_result == result_to_check)


def test_push_project_operator():
    sink_operator = Sink(inputs=[])
    project_operator = Project(inputs=[], outputs=[sink_operator], fields_to_keep=["UID2"]) #"UID1",
    scan_operator_1 = Scan(filepath=friends_relation, outputs=[project_operator])

    scan_operator_1.start()
    
    correct_result = [(2,), (3,), (4,), (5,)]
    sink_output = sink_operator.output_data
    
    result_to_check = []
    for item in sink_output:
        result_to_check.append(item.tuple)
    
    assert(correct_result == result_to_check)






def test_query_1():
    assert(1 == 1)