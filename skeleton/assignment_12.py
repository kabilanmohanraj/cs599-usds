from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
import argparse
from ast import alias
from audioop import reverse
import collections

import csv
from email import header
import logging
from enum import Enum
import math
import operator
from statistics import mean
from turtle import right
from typing import List, Tuple
from unittest import case
import uuid
# from numpy import average
from itertools import count, groupby

# from yaml import scan

# import ray

# Note (john): Make sure you use Python's logger to log
#              information about your program
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# Generates unique operator IDs
def _generate_uuid():
    return uuid.uuid4()

# Partition strategy enum
class PartitionStrategy(Enum):
    RR = "Round_Robin"
    HASH = "Hash_Based"

# Custom tuple class with optional metadata
class ATuple:
    """Custom tuple.

    Attributes:
        tuple (Tuple): The actual tuple.
        metadata (string): The tuple metadata (e.g. provenance annotations).
        operator (Operator): A handle to the operator that produced the tuple.
    """
    def __init__(self, tuple, metadata=None, operator=None):
        self.tuple = tuple
        self.metadata = metadata
        self.operator = operator

    # Returns the lineage of self
    def lineage(self) -> List[ATuple]:
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the Where-provenance of the attribute at index 'att_index' of self
    def where(self, att_index) -> List[Tuple]:
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

    # Returns the How-provenance of self
    def how(self) -> str:
        # YOUR CODE HERE (ONLY FOR TASK 3 IN ASSIGNMENT 2)
        pass

    # Returns the input tuples with responsibility \rho >= 0.5 (if any)
    def responsible_inputs(self) -> List[Tuple]:
        # YOUR CODE HERE (ONLY FOR TASK 4 IN ASSIGNMENT 2)
        pass

# Data operator
class Operator:
    """Data operator (parent class).

    Attributes:
        id (string): Unique operator ID.
        name (string): Operator name.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
        pull (bool): Defines whether to use pull-based (True) vs
        push-based (False) evaluation.
        partition_strategy (Enum): Defines the output partitioning
        strategy.
    """
    def __init__(self,
                 id=None,
                 name=None,
                 track_prov=False,
                 propagate_prov=False,
                 pull=True,
                 partition_strategy : PartitionStrategy = PartitionStrategy.RR):
        self.id = _generate_uuid() if id is None else id
        self.name = "Undefined" if name is None else name
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.pull = pull
        self.partition_strategy = partition_strategy
        logger.debug("Created {} operator with id {}".format(self.name,
                                                             self.id))

    # NOTE (john): Must be implemented by the subclasses
    def get_next(self) -> List[ATuple]:
        logger.error("Method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def lineage(self, tuples: List[ATuple]) -> List[List[ATuple]]:
        logger.error("Lineage method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def where(self, att_index: int, tuples: List[ATuple]) -> List[List[Tuple]]:
        logger.error("Where-provenance method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def apply(self, tuples: List[ATuple]) -> bool:
        logger.error("Apply method is not implemented!")

# Scan operator
class Scan(Operator):
    """Scan operator.

    Attributes:
        filepath (string): The path to the input file.
        outputs (List): A list of handles to the instances of the next
        operator in the plan.
        filter (function): An optional user-defined filter.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
        pull (bool): Defines whether to use pull-based (True) vs
        push-based (False) evaluation.
        partition_strategy (Enum): Defines the output partitioning
        strategy.
    """
    scan_operator_number = 0 # The variable is used in the indexing scheme for the attributes
    column_headers_to_index = {} # Mapping attributes to numbers

    # Initializes scan operator
    def __init__(self,
                 filepath,
                 outputs : List[Operator],
                 filter=None,
                 track_prov=False,
                 propagate_prov=False,
                 pull=True,
                 partition_strategy : PartitionStrategy = PartitionStrategy.RR,
                 batch_size=100000,
                 relation_tag="L"):
        super(Scan, self).__init__(name="Scan",
                                   track_prov=track_prov,
                                   propagate_prov=propagate_prov,
                                   pull=pull,
                                   partition_strategy=partition_strategy)
        # YOUR CODE HERE
        self.filepath = filepath
        self.outputs = outputs
        self.filter = filter
        self.batch_size = batch_size
        self.scan_pointer = 1

        self.relation_tag = relation_tag
        self.column_headers = []

        with open(self.filepath) as input_file:
            input_reader = csv.reader(input_file, delimiter=" ")
            self.column_headers = input_reader.__next__()[1:]
        
        # for i in range(len(column_headers)): # Assign index to the column headers
        #     Scan.column_headers_to_index[column_headers[i]] = str(Scan.scan_operator_number)+"_"+str(i)
        # Scan.scan_operator_number += 1

    # Returns next batch of tuples in given file (or None if file exhausted)
    def get_next(self):
        output_data = []

        with open(self.filepath) as input_file:
            output_data = [ATuple(tuple = tuple(map(int, row.split(" ")))) for idx, row in enumerate(input_file) if idx in range(self.scan_pointer, self.scan_pointer + self.batch_size)]

        self.scan_pointer += self.batch_size
        
        if(output_data != []):
            annotated_output = [self.column_headers, output_data]
        else:
            annotated_output = [self.column_headers, None]
        
        return annotated_output

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

    # Starts the process of reading tuples (only for push-based evaluation)
    def start(self):
        output_data = []
        while(True):
            with open(self.filepath) as input_file:
                output_data = [ATuple(tuple = tuple(map(int, row.split(" ")))) for idx, row in enumerate(input_file) if idx in range(self.scan_pointer, self.scan_pointer + self.batch_size)]

            self.scan_pointer += self.batch_size

            annotated_data = [self.column_headers, output_data, self.relation_tag] # the list contains information about the column headers in all the relations and information about the source of a relation (Left / Right)

            if(output_data != []):
                self.outputs[0].apply(annotated_data)
            else:
                self.outputs[0].apply(None)
                break

            

# Equi-join operator
class Join(Operator):
    """Equi-join operator.

    Attributes:
        left_inputs (List): A list of handles to the instances of the operator
        that produces the left input.
        right_inputs (List):A list of handles to the instances of the operator
        that produces the right input.
        outputs (List): A list of handles to the instances of the next
        operator in the plan.
        left_join_attribute (int): The index of the left join attribute.
        right_join_attribute (int): The index of the right join attribute.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
        pull (bool): Defines whether to use pull-based (True) vs
        push-based (False) evaluation.
        partition_strategy (Enum): Defines the output partitioning
        strategy.
    """
    # Initializes join operator
    def __init__(self,
                 left_inputs : List[Operator],
                 right_inputs : List[Operator],
                 outputs : List[Operator],
                 left_join_attribute,
                 right_join_attribute,
                 track_prov=False,
                 propagate_prov=False,
                 pull=True,
                 partition_strategy : PartitionStrategy = PartitionStrategy.RR):
        super(Join, self).__init__(name="Join",
                                   track_prov=track_prov,
                                   propagate_prov=propagate_prov,
                                   pull=pull,
                                   partition_strategy=partition_strategy)
        # YOUR CODE HERE
        self.left_inputs = left_inputs
        self.right_inputs = right_inputs
        self.left_join_attribute = left_join_attribute
        self.right_join_attribute = right_join_attribute
        self.outputs = outputs

        self.left_relation_hash = collections.defaultdict(list)
        self.right_relation_hash = collections.defaultdict(list)
        self.table_to_hash = 0

        self.column_headers = []

        self.is_first_call_left = True
        self.is_first_call_right = True

    # Returns next batch of joined tuples (or None if done)
    def get_next(self):
        output_from_probing = []

        if(self.is_first_call_left):
            self.is_first_call_left = False
            for operator in self.left_inputs:
                while(True):
                    tuples = operator.get_next()
                    if(tuples[1] is not None):
                        # Collect the left relation into a dictionary (hashing) only on the first call
                        for atup_left in tuples[1]:
                            self.left_relation_hash[str(atup_left.tuple[self.left_join_attribute])].append(atup_left)
                    else:
                        self.column_headers += tuples[0]
                        break
            
            # logger.debug(self.left_relation_hash["1769"]) # returns empty list if key not found
            # logger.info("Completed hashing phase. Starting probing phase====")
        for operator in self.right_inputs:
            tuples = operator.get_next()
            if(self.is_first_call_right):
                self.column_headers += tuples[0]
                self.is_first_call_right = False
            if(tuples[1] is None):
                break
            # Collect the right relation in batches and compare with the previously created dictionary (probing)
            
            for atup_right in tuples[1]:
                matching_key = self.left_relation_hash[str(atup_right.tuple[self.right_join_attribute])]
                if(matching_key != []):
                    for element in matching_key:
                        output_from_probing.append((element, atup_right))

        if(tuples[1] is None):
            annotated_output = [self.column_headers, None]
        else:
            annotated_output = [self.column_headers, output_from_probing]

        return annotated_output

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        output_from_probing = []

        if(tuples is not None):
            # split tuples L and R
            if(tuples[2] == "L"): # Probe the right relation hash and then hash the left relation
                if(self.is_first_call_left):
                    self.column_headers += tuples[0]
                    self.is_first_call_left = False
                for tup in tuples[1]:
                    probing_key = str(tup.tuple[self.left_join_attribute])
                    if(probing_key in self.right_relation_hash.keys()):
                        matching_key = self.right_relation_hash[probing_key]
                        if(matching_key != []):
                            for element in matching_key:
                                output_from_probing.append((element, tup))
                    self.left_relation_hash[str(tup.tuple[self.left_join_attribute])].append(tup)

            if(tuples[2] == "R"):
                if(self.is_first_call_right):
                    self.column_headers += tuples[0]
                    self.is_first_call_right = False
                for tup in tuples[1]:
                    probing_key = str(tup.tuple[self.right_join_attribute])
                    if(probing_key in self.left_relation_hash.keys()):
                        matching_key = self.left_relation_hash[probing_key]
                        if(matching_key != []):
                            for element in matching_key:
                                output_from_probing.append((element, tup))
                self.right_relation_hash[str(tup.tuple[self.right_join_attribute])].append(tup)

            annotated_output = [self.column_headers, output_from_probing]
            self.outputs[0].apply(annotated_output)

            return
        else:
            self.outputs[0].apply(None)



# Project operator
class Project(Operator):
    """Project operator.

    Attributes:
        inputs (List): A list of handles to the instances of the previous
        operator in the plan.
        outputs (List): A list of handles to the instances of the next
        operator in the plan.
        fields_to_keep (List(int)): A list of attribute indices to keep.
        If empty, the project operator behaves like an identity map, i.e., it
        produces and output that is identical to its input.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
        pull (bool): Defines whether to use pull-based (True) vs
        push-based (False) evaluation.
        partition_strategy (Enum): Defines the output partitioning
        strategy.
    """
    # Initializes project operator
    def __init__(self,
                 inputs : List[Operator],
                 outputs : List[None],
                 fields_to_keep=[],
                 aliasing=False,
                 track_prov=False,
                 propagate_prov=False,
                 pull=True,
                 partition_strategy : PartitionStrategy = PartitionStrategy.RR):
        super(Project, self).__init__(name="Project",
                                      track_prov=track_prov,
                                      propagate_prov=propagate_prov,
                                      pull=pull,
                                      partition_strategy=partition_strategy)
        # YOUR CODE HERE
        self.inputs = inputs
        self.outputs = outputs
        self.fields_to_keep = fields_to_keep
        self.aliasing = aliasing

        self.column_headers = []

    # Return next batch of projected tuples (or None if done)
    def get_next(self):
        projected_column = []
        project_fields_to_index = []
        
        for operator in self.inputs:
            tuples = operator.get_next()
            self.column_headers = tuples[0]
            if(self.aliasing):
                annotated_output = [self.fields_to_keep, tuples[1]]
                return annotated_output
            else:
                if(tuples[1] is not None):
                    for field in self.fields_to_keep:
                        if field in self.column_headers:
                            project_fields_to_index.append(self.column_headers.index(field))

                    try: # For nested tuples
                        for element in tuples[1]:
                            flattened_tuple = tuple()
                            for item in element:
                                for sub_item in item.tuple:
                                    flattened_tuple += (sub_item,)

                            temp = []
                            for index in project_fields_to_index:
                                temp.append(flattened_tuple[index])
                            projected_column.append(ATuple(tuple(temp)))
                    except:
                        for element in tuples[1]: # For flat tuples
                            temp = []
                            for index in project_fields_to_index:
                                temp.append(element.tuple[index])
                            projected_column.append(ATuple(tuple(temp)))
        
        if(tuples[1] is None): 
            return [self.fields_to_keep, None]
        else:
            return [self.fields_to_keep, projected_column]

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]): # add column_headers_to_index in the function call from join
        if(self.aliasing):
            annotated_output = [self.fields_to_keep, tuples[1]]
            self.outputs[0].apply(annotated_output)
        else:
            if(tuples is not None):
                projected_column = []
                project_fields_to_index = []
                self.column_headers = tuples[0]
                for field in self.fields_to_keep:
                    if field in self.column_headers:
                        project_fields_to_index.append(self.column_headers.index(field))
                
                if(tuples[1] != []):
                    try: # For nested tuples
                        for element in tuples[1]:
                            flattened_tuple = tuple()
                            for item in element:
                                for sub_item in item.tuple:
                                    flattened_tuple += (sub_item,)

                            temp = []
                            for index in project_fields_to_index:
                                temp.append(flattened_tuple[index])
                            projected_column.append(ATuple(tuple(temp)))
                    except: # For flat tuples)
                        for element in tuples[1]: 
                            temp = []
                            for index in project_fields_to_index:
                                temp.append(element.tuple[index])
                            projected_column.append(ATuple(tuple(temp)))
                            
                    
                    annotated_output = [self.fields_to_keep, projected_column]
                    self.outputs[0].apply(annotated_output)

                else:
                    return
            else:
                self.outputs[0].apply(None)


# Group-by operator
class GroupBy(Operator):
    """Group-by operator.

    Attributes:
        inputs (List): A list of handles to the instances of the previous
        operator in the plan.
        outputs (List): A list of handles to the instances of the next
        operator in the plan.
        key (int): The index of the key to group tuples.
        value (int): The index of the attribute we want to aggregate.
        agg_fun (function): The aggregation function (e.g. AVG)
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
        pull (bool): Defines whether to use pull-based (True) vs
        push-based (False) evaluation.
        partition_strategy (Enum): Defines the output partitioning
        strategy.
    """
    # Initializes average operator
    def __init__(self,
                 inputs : List[Operator],
                 outputs : List[Operator],
                 key,
                 value,
                 agg_fun,
                 track_prov=False,
                 propagate_prov=False,
                 pull=True,
                 partition_strategy : PartitionStrategy = PartitionStrategy.RR):
        super(GroupBy, self).__init__(name="GroupBy",
                                      track_prov=track_prov,
                                      propagate_prov=propagate_prov,
                                      pull=pull,
                                      partition_strategy=partition_strategy)
        # YOUR CODE HERE
        self.inputs = inputs
        self.outputs = outputs
        self.agg_fun = agg_fun
        self.key = key
        self.value = value

        self.column_headers = []
        self.grouping_dict = collections.defaultdict(list)
        self.is_first_none = True

    # Returns aggregated value per distinct key in the input (or None if done)
    def get_next(self):
        output_data = []
        output_dict = {}
        for operator in self.inputs:
            while(True):
                tuples = operator.get_next()
                if(tuples[1] is None): # is None
                    break
                for item in tuples[1]:
                    if(self.key == 0 and self.value == 0):
                        self.grouping_dict[str(self.key)].append(int(item.tuple[self.value]))
                    else:
                        self.grouping_dict[str(item.tuple[self.key])].append(int(item.tuple[self.value]))

            if(self.key == 0 and self.value == 0):
                output_data += [
                        ATuple(
                            (self.agg_fun(self.grouping_dict[str(self.key)]),)
                            )
                        ]
            else:
                for dict_item in enumerate(self.grouping_dict.items()):
                    output_dict[dict_item[1][self.key]] = self.agg_fun(dict_item[1][self.value])
                    output_data.append(ATuple((int(dict_item[1][self.key]), output_dict[dict_item[1][self.key]])))
        
        annotated_output = [tuples[0], output_data]
        return annotated_output

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        output_data = []
        if(tuples is not None):
            self.column_headers = tuples[0]
            if(tuples[1] != []):
                for item in tuples[1]:
                    if(self.key == 0 and self.value == 0):
                        self.grouping_dict[str(self.key)].append(int(item.tuple[self.value]))
                    else:
                        self.grouping_dict[str(item.tuple[self.key])].append(int(item.tuple[self.value]))
            else:
                return
        else:
            if(self.is_first_none):
                self.is_first_none = False
            else:
                output_dict = {}
                if(self.key == 0 and self.value == 0):
                    output_data += [
                        ATuple(
                            (self.agg_fun(self.grouping_dict[str(self.key)]),)
                            )
                        ]
                else:
                    for dict_item in enumerate(self.grouping_dict.items()):
                        output_dict[dict_item[1][self.key]] = self.agg_fun(dict_item[1][self.value])
                        output_data.append(ATuple((int(dict_item[1][self.key]), output_dict[dict_item[1][self.key]])))

                annotated_output = [self.column_headers, output_data]
                if(self.outputs != []):
                    self.outputs[0].apply(annotated_output)


# Custom histogram operator
class Histogram(Operator):
    """Histogram operator.

    Attributes:
        inputs (List): A list of handles to the instances of the previous
        operator in the plan.
        outputs (List): A list of handles to the instances of the next
        operator in the plan.
        key (int): The index of the key to group tuples. The operator outputs
        the total number of tuples per distinct key.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
        pull (bool): Defines whether to use pull-based (True) vs
        push-based (False) evaluation.
        partition_strategy (Enum): Defines the output partitioning
        strategy.
    """
    # Initializes histogram operator
    def __init__(self,
                 inputs : List[Operator],
                 outputs : List[Operator],
                 key=0,
                 track_prov=False,
                 propagate_prov=False,
                 pull=True,
                 partition_strategy : PartitionStrategy = PartitionStrategy.RR):
        super(Histogram, self).__init__(name="Histogram",
                                        track_prov=track_prov,
                                        propagate_prov=propagate_prov,
                                        pull=pull,
                                        partition_strategy=partition_strategy)
        
        self.inputs = inputs
        self.outputs = outputs
        self.key = key

        self.input_tuples = []
        self.grouping_dict = collections.defaultdict(list)
        self.column_headers = []

    # Returns histogram (or None if done)
    def get_next(self):
        output_data = []
        output_dict = {}
        for operator in self.inputs:
            while(True):
                tuples = operator.get_next()
                if(tuples[1] is None):
                    break
                for item in tuples[1]:
                    self.grouping_dict[str(item.tuple[self.key])].append(1)
                        
            for dict_item in enumerate(self.grouping_dict.items()):
                output_dict[dict_item[1][self.key]] = sum(dict_item[1][1])
                output_data.append(ATuple((int(dict_item[1][self.key]), output_dict[dict_item[1][self.key]])))

        annotated_output = [tuples[0], output_data]
        return annotated_output

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        if(tuples is not None):
            self.column_headers = tuples[0]
            if(tuples[1] != []):
                for item in tuples[1]:
                    self.grouping_dict[str(item.tuple[self.key])].append(1)
            else:
                return
        else:
            output_dict = {}
            output_data = []
            for dict_item in enumerate(self.grouping_dict.items()):
                output_dict[dict_item[1][self.key]] = sum(dict_item[1][1])
                output_data.append(ATuple((int(dict_item[1][self.key]), output_dict[dict_item[1][self.key]])))

            annotated_output = [self.column_headers, output_data]
            if(self.outputs != []):
                self.outputs[0].apply(annotated_output)

            

# Order by operator
class OrderBy(Operator):
    """OrderBy operator.

    Attributes:
        inputs (List): A list of handles to the instances of the previous
        operator in the plan.
        outputs (List): A list of handles to the instances of the next
        operator in the plan.
        comparator (function): The user-defined comparator used for sorting the
        input tuples.
        ASC (bool): True if sorting in ascending order, False otherwise.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
        pull (bool): Defines whether to use pull-based (True) vs
        push-based (False) evaluation.
        partition_strategy (Enum): Defines the output partitioning
        strategy.
    """
    # Initializes order-by operator
    def __init__(self,
                 inputs : List[Operator],
                 outputs : List[Operator],
                 comparator,
                 ASC=True,
                 track_prov=False,
                 propagate_prov=False,
                 pull=True,
                 partition_strategy : PartitionStrategy = PartitionStrategy.RR):
        super(OrderBy, self).__init__(name="OrderBy",
                                      track_prov=track_prov,
                                      propagate_prov=propagate_prov,
                                      pull=pull,
                                      partition_strategy=partition_strategy)
        
        self.inputs = inputs
        self.outputs = outputs
        self.comparator = comparator
        self.ASC = ASC

    # Returns the sorted input (or None if done)
    def get_next(self):
        for operator in self.inputs:
            tuples = operator.get_next()
            tuple_list = [value.tuple for value in tuples[1]]
            # Sort list by the first element
            sorted_list = self.comparator(tuple_list, key=lambda x:x[0])
            output_data = [ATuple(item) for item in sorted_list]
            annotated_output = [tuples[0], output_data]

        return annotated_output


    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        tuple_list = [value.tuple for value in tuples[1]]
        if(self.ASC):
            reversed = False
        else:
            reversed = True
        # Sort list by the second element
        sorted_list = self.comparator(tuple_list, key=lambda x:x[1], reverse=reversed)
        output_data = [ATuple(item) for item in sorted_list]
        annotated_output = [tuples[0], output_data]
        if(self.outputs != []):
            self.outputs[0].apply(annotated_output)


# Top-k operator
class TopK(Operator):
    """TopK operator.

    Attributes:
        inputs (List): A list of handles to the instances of the previous
        operator in the plan.
        outputs (List): A list of handles to the instances of the next
        operator in the plan.
        k (int): The maximum number of tuples to output.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
        pull (bool): Defines whether to use pull-based (True) vs
        push-based (False) evaluation.
        partition_strategy (Enum): Defines the output partitioning
        strategy.
    """
    # Initializes top-k operator
    def __init__(self,
                 inputs : List[Operator],
                 outputs : List[Operator],
                 k=None,
                 track_prov=False,
                 propagate_prov=False,
                 pull=True,
                 partition_strategy : PartitionStrategy = PartitionStrategy.RR):
        super(TopK, self).__init__(name="TopK",
                                   track_prov=track_prov,
                                   propagate_prov=propagate_prov,
                                   pull=pull,
                                   partition_strategy=partition_strategy)
       
        self.inputs = inputs
        self.outputs = outputs
        self.k = k

    # Returns the first k tuples in the input (or None if done)
    def get_next(self):
        for operator in self.inputs:
            tuples = operator.get_next()
            output_data = tuples[1][0:self.k]
            annotated_output = [tuples[0], output_data]
        
            return annotated_output

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 1 IN ASSIGNMENT 2)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        if(self.outputs != []):
            annotated_output = [tuples[0], tuples[1][0:self.k]]
            self.outputs[0].apply(annotated_output)


# Filter operator
class Select(Operator):
    """Select operator.

    Attributes:
        inputs (List): A list of handles to the instances of the previous
        operator in the plan.
        outputs (List): A list of handles to the instances of the next
        operator in the plan.
        predicate (function): The selection predicate.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
        pull (bool): Defines whether to use pull-based (True) vs
        push-based (False) evaluation.
        partition_strategy (Enum): Defines the output partitioning
        strategy.
    """
    # Initializes select operator
    def __init__(self,
                 inputs : List[Operator],
                 outputs : List[Operator],
                 predicate,
                 track_prov=False,
                 propagate_prov=False,
                 pull=True,
                 partition_strategy : PartitionStrategy = PartitionStrategy.RR):
        super(Select, self).__init__(name="Select",
                                     track_prov=track_prov,
                                     propagate_prov=propagate_prov,
                                     pull=pull,
                                     partition_strategy=partition_strategy)
        self.inputs = inputs
        self.outputs = outputs
        self.predicate = predicate

    # Returns next batch of tuples that pass the filter (or None if done)
    def get_next(self):
        for operator in self.inputs:
            tuples = operator.get_next()
            if(self.predicate is None):
                return tuples
            filtered_output_data = self.predicate(tuples[1])
        if(tuples[1] is None):
            return [tuples[0], None]
        else:
            return [tuples[0], filtered_output_data]

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        if(self.predicate is None):
            return tuples
        if(tuples is not None):
            filtered_output_data = self.predicate(tuples[1])
            tuples[1] = filtered_output_data.copy()
            self.outputs[0].apply(tuples)
            return
        else:
            self.outputs[0].apply(None)



# Sink operator
class Sink(Operator):
    """Sink operator.

    Attributes:
        inputs (List): A list of handles to the instances of the previous
        operator in the plan.
        outputs (List): A list of handles to the instances of the next
        operator in the plan.
        filepath (str): The path to the output file.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
        pull (bool): Defines whether to use pull-based (True) vs
        push-based (False) evaluation.
        partition_strategy (Enum): Defines the output partitioning
        strategy.
    """
    # Initializes select operator
    def __init__(self,
                 inputs : List[Operator],
                 outputs : List[Operator],
                 filepath="./query_output.txt",
                 track_prov=False,
                 propagate_prov=False,
                 pull=True,
                 partition_strategy : PartitionStrategy = PartitionStrategy.RR):
        super(Sink, self).__init__(name="Sink",
                                     track_prov=track_prov,
                                     propagate_prov=propagate_prov,
                                     pull=pull,
                                     partition_strategy=partition_strategy)
        self.inputs = inputs
        self.outputs = outputs
        self.filepath = filepath

    # Returns next batch of tuples that pass the filter (or None if done)
    def get_next(self):
        for operator in self.inputs:
            tuples = operator.get_next()
            self.write_to_csv(tuples)
        
    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        self.write_to_csv(tuples)
    
    # Writes the tuples to the CSV file with column headers
    def write_to_csv(self, tuples):
        with open(self.filepath, 'w') as output_file:
            output_writer = csv.writer(output_file, delimiter=",")
            output_writer.writerow(tuples[0]) # Write column headers
            for item in tuples[1]:
                output_writer.writerow(item.tuple)



## Filter functions to be passed to the "Select" operator           
def relation_filter_left(scan_output_left):
    if(scan_output_left is not None):
        filter_output_left = [value for value in scan_output_left if(value.tuple[0] == user_id)]
        return filter_output_left

def relation_filter_right(scan_output_right):
    if(scan_output_right is not None):
        filter_output_right = [value for value in scan_output_right if(value.tuple[1] == movie_id)]
        return filter_output_right
    


## Driver
if __name__ == "__main__":

    logger.info("Assignment #1")

    # Initialize parser
    parser = argparse.ArgumentParser()

    options_list = ["query", "ff", "mf", "uid", "mid", "pull", "output"]
    # "pull", "output"

    for argument in options_list:
        parser.add_argument("--"+argument)

    args = parser.parse_args()

    # TASK 1: Implement 'likeness' prediction query for User A and Movie M
    #
    # SELECT AVG(R.Rating)
    # FROM Friends as F, Ratings as R
    # WHERE F.UID2 = R.UID
    #       AND F.UID1 = 'A'
    #       AND R.MID = 'M'

    # YOUR CODE HERE
    movie_id = int(args.mid)
    user_id = int(args.uid)

    relation_1 = args.ff
    relation_2 = args.mf

    if(args.query == "1"):
        if(args.pull == "1"):
            ## -------------------------
            ## Pull-based
            ## -------------------------
            scan_operator_left = Scan(filepath=relation_1, outputs=[], relation_tag="L")
            scan_operator_right = Scan(filepath=relation_2, outputs=[], relation_tag="R")

            filter_operator_left = Select(inputs=[scan_operator_left], outputs=[], predicate=relation_filter_left)
            filter_operator_right = Select(inputs=[scan_operator_right], outputs=[], predicate=relation_filter_right)

            join_operator = Join(left_inputs=[filter_operator_left], right_inputs=[filter_operator_right], outputs=[], left_join_attribute=1, right_join_attribute=0)

            project_operator = Project(inputs=[join_operator], outputs=[], fields_to_keep=["Rating"])

            average_operator = GroupBy(inputs=[project_operator], outputs=[], agg_fun=mean, key=0, value=0)

            sink_operator = Sink(inputs=[average_operator], outputs=[], filepath=args.output)
            sink_operator.get_next()

        else:
            ## -------------------------
            ## Push-based
            ## -------------------------
            sink_operator = Sink(inputs=[], outputs=[], filepath=args.output)

            average_operator = GroupBy(inputs=[], outputs=[sink_operator], agg_fun=mean, key=0, value=0) # key and value are attribute numbers after projection
            # key = -1 means only 1 column was projected

            project_operator = Project(inputs=[], outputs=[average_operator], fields_to_keep=["Rating"])

            join_operator = Join(left_inputs=[], right_inputs=[], outputs=[project_operator], left_join_attribute=1, right_join_attribute=0)

            filter_operator_left = Select(inputs=[], outputs=[join_operator], predicate=relation_filter_left)
            filter_operator_right = Select(inputs=[], outputs=[join_operator], predicate=relation_filter_right)  

            scan_operator_left = Scan(filepath=relation_1, outputs=[filter_operator_left], relation_tag="L")
            scan_operator_right = Scan(filepath=relation_2, outputs=[filter_operator_right], relation_tag="R")

            scan_operator_left.start()
            scan_operator_right.start()
    


    # TASK 2: Implement recommendation query for User A
    #
    # SELECT R.MID
    # FROM ( SELECT R.MID, AVG(R.Rating) as score
    #        FROM Friends as F, Ratings as R
    #        WHERE F.UID2 = R.UID
    #              AND F.UID1 = 'A'
    #        GROUP BY R.MID
    #        ORDER BY score DESC
    #        LIMIT 1 )

    # YOUR CODE HERE
    if(args.query == "2"):
        if(args.pull == "1"):
            ## -------------------------
            ## Pull-based
            ## -------------------------
            scan_operator_left = Scan(filepath=relation_1, outputs=[], relation_tag="L")
            scan_operator_right = Scan(filepath=relation_2, outputs=[], relation_tag="R")

            filter_operator_left = Select(inputs=[scan_operator_left], outputs=[], predicate=relation_filter_left)

            join_operator = Join(left_inputs=[filter_operator_left], right_inputs=[scan_operator_right], outputs=[], left_join_attribute=1, right_join_attribute=0)

            project_operator = Project(inputs=[join_operator], outputs=[], fields_to_keep=["MID", "Rating"])

            groupby_operator = GroupBy(inputs=[project_operator], outputs=[], agg_fun=mean, key=0, value=1)

            orderby_operator = OrderBy(inputs=[groupby_operator], outputs=[], comparator=sorted, ASC=False)

            limit_operator = TopK(inputs=[orderby_operator], outputs=[], k=1)

            project_operator = Project(inputs=[limit_operator], outputs=[], fields_to_keep=["MID"], aliasing=False)

            sink_operator = Sink(inputs=[project_operator], outputs=[], filepath=args.output)
            sink_operator.get_next()

        else:
            ## -------------------------
            ## Push-based
            ## -------------------------
            sink_operator = Sink(inputs=[], outputs=[], filepath=args.output)

            project_operator = Project(inputs=[], outputs=[sink_operator], fields_to_keep=["MID"], aliasing=False)

            limit_operator = TopK(inputs=[], outputs=[project_operator], k=1)

            orderby_operator = OrderBy(inputs=[], outputs=[limit_operator], comparator=sorted, ASC=False)

            groupby_operator = GroupBy(inputs=[], outputs=[orderby_operator], agg_fun=mean, key=0, value=1) # key and value are attribute numbers after projection
            # key = -1 means only 1 column was projected

            project_operator = Project(inputs=[], outputs=[groupby_operator], fields_to_keep=["MID", "Rating"])

            join_operator = Join(left_inputs=[], right_inputs=[], outputs=[project_operator], left_join_attribute=1, right_join_attribute=0)

            filter_operator_left = Select(inputs=[], outputs=[join_operator], predicate=relation_filter_left) 

            scan_operator_left = Scan(filepath=relation_1, outputs=[filter_operator_left], relation_tag="L")
            scan_operator_right = Scan(filepath=relation_2, outputs=[join_operator], relation_tag="R")

            scan_operator_left.start()
            scan_operator_right.start()


    # TASK 3: Implement explanation query for User A and Movie M
    #
    # SELECT HIST(R.Rating) as explanation
    # FROM Friends as F, Ratings as R
    # WHERE F.UID2 = R.UID
    #       AND F.UID1 = 'A'
    #       AND R.MID = 'M'

    if(args.query == "3"):
        if(args.pull == "1"):
            scan_operator_left = Scan(filepath=relation_1, outputs=[], relation_tag="L")
            scan_operator_right = Scan(filepath=relation_2, outputs=[], relation_tag="R")

            filter_operator_left = Select(inputs=[scan_operator_left], outputs=[], predicate=relation_filter_left)
            filter_operator_right = Select(inputs=[scan_operator_right], outputs=[], predicate=relation_filter_right)

            join_operator = Join(left_inputs=[filter_operator_left], right_inputs=[filter_operator_right], outputs=[], left_join_attribute=1, right_join_attribute=0)

            project_operator = Project(inputs=[join_operator], outputs=[], fields_to_keep=["Rating"])

            histogram_operator = Histogram(inputs=[project_operator], outputs=[], key=0)

            project_operator = Project(inputs=[histogram_operator], outputs=[], fields_to_keep=["Rating", "Count"], aliasing=True)

            sink_operator = Sink(inputs=[project_operator], outputs=[], filepath=args.output)
            sink_operator.get_next()
        else:
            ## -------------------------
            ## Push-based
            ## -------------------------
            sink_operator = Sink(inputs=[], outputs=[], filepath=args.output)

            project_operator = Project(inputs=[], outputs=[sink_operator], fields_to_keep=["Rating", "Count"], aliasing=True)

            histogram_operator = Histogram(inputs=[], outputs=[project_operator], key=0) # key is the attribute index after projection
            # key = -1 means only 1 column was projected

            project_operator = Project(inputs=[], outputs=[histogram_operator], fields_to_keep=["Rating"])

            join_operator = Join(left_inputs=[], right_inputs=[], outputs=[project_operator], left_join_attribute=1, right_join_attribute=0)

            filter_operator_left = Select(inputs=[], outputs=[join_operator], predicate=relation_filter_left)
            filter_operator_right = Select(inputs=[], outputs=[join_operator], predicate=relation_filter_right)  

            scan_operator_left = Scan(filepath=relation_1, outputs=[filter_operator_left], relation_tag="L")
            scan_operator_right = Scan(filepath=relation_2, outputs=[filter_operator_right], relation_tag="R")

            scan_operator_left.start()
            scan_operator_right.start()


    # TASK 4: Turn your data operators into Ray actors
    #
    # NOTE (john): Add your changes for Task 4 to a new git branch 'ray'


    logger.info("Assignment #2")

    # TASK 1: Implement lineage query for movie recommendation

    # YOUR CODE HERE


    # TASK 2: Implement where-provenance query for 'likeness' prediction

    # YOUR CODE HERE


    # TASK 3: Implement how-provenance query for movie recommendation

    # YOUR CODE HERE


    # TASK 4: Retrieve most responsible tuples for movie recommendation

    # YOUR CODE HERE
