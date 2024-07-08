from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
import argparse
import collections

import csv
from email import header
import logging
from enum import Enum
from operator import itemgetter
import re
from statistics import mean
from typing import List, Tuple
import uuid


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
    def __init__(self, tuple, metadata=None, operator=None, tuple_identifier=None):
        self.tuple = tuple
        self.metadata = metadata
        self.operator = operator
        if tuple_identifier is None:
            self.tuple_identifier = _generate_uuid()
        else:
            self.tuple_identifier = tuple_identifier
    
    # Returns the lineage of self
    def lineage(self) -> List[ATuple]:
        return self.operator.lineage(tuples=[self])

    def __hash__(self):
        return int(self.tuple_identifier)

    # Returns the Where-provenance of the attribute at index 'att_index' of self
    def where(self, att_index) -> List[Tuple]:
        return self.operator.where(att_index, tuples=[self])

    # Returns the How-provenance of self
    def how(self) -> str:
        return self.metadata

    # Returns the input tuples with responsibility \rho >= 0.5 (if any)
    def responsible_inputs(self) -> List[Tuple]:
        temp_dict = GroupBy.how_provenance_dict
        print(temp_dict)
        extracted_tuple_strings = []
        output_tuples = temp_dict.keys()
        for element in temp_dict.values():
            temp_string = re.sub(r"([A-Z(),])", "", element)
            temp_string = temp_string.strip().split(" ")
            extracted_tuple_strings.append(temp_string)

        string_to_tuples = [] # Nested list
        flat_tuples = [] # Flat list
        for element in extracted_tuple_strings:
            temp_2 = []
            for subelement in element:
                temp = []
                temp_1 = []
                temp = re.split('@|\*', subelement)
                temp_1.append(Scan.how_identifier_to_tuple_map[temp[0]])
                temp_1.append(Scan.how_identifier_to_tuple_map[temp[1]])
                flat_tuples += temp_1
                temp_1.append(int(temp[2]))
                temp_2.append(temp_1)
            string_to_tuples.append(temp_2)
        
        counterfactual_causes = set()
        actual_causes = set()

        def recalculate_agg_value(ignore_tuples):
            agg_value_list = []
            for element in string_to_tuples:
                element_count = 0
                element_sum = 0
                for subelement in element:
                    if(isinstance(ignore_tuples, ATuple)):
                        if(ignore_tuples in subelement):
                            continue
                    if(isinstance(ignore_tuples, list)):
                        if(ignore_tuples[0] in subelement or ignore_tuples[1] in subelement):
                            continue
                    element_sum += subelement[2]
                    element_count += 1

                try:
                    agg_value_list.append(element_sum/element_count)
                except ZeroDivisionError:
                    agg_value_list.append(0)

            return agg_value_list

        original_agg_values = list(map(itemgetter(1), list(output_tuples)))
        original_topk_agg_value = max(original_agg_values)
        original_topk_indices = set()
        new_agg_values = []

        i = 0
        while(i < len(original_agg_values)):
            if(original_agg_values[i] == original_topk_agg_value):
                original_topk_indices.add(i)
            i += 1

        # Finding counterfactual causes
        i = 0
        while(i < len(flat_tuples)):
            new_agg_values = recalculate_agg_value(flat_tuples[i])
            new_topk_agg_value = max(new_agg_values)
            new_topk_indices = set()
            j = 0
            while(j < len(new_agg_values)):
                if(new_agg_values[j] == new_topk_agg_value):
                    new_topk_indices.add(j)
                j += 1
            if(not original_topk_indices.issubset(new_topk_indices)):
                counterfactual_causes.add((flat_tuples[i].tuple, 1,))
            i += 1

        # Finding actual causes
        i = 0
        while(i < len(flat_tuples)):
            j = 0
            while(j < len(flat_tuples)):
                if(flat_tuples[i].tuple not in counterfactual_causes and flat_tuples[j].tuple not in counterfactual_causes):
                    new_agg_values = recalculate_agg_value([flat_tuples[i], flat_tuples[j]])
                    new_topk_agg_value = max(new_agg_values)
                    new_topk_indices = set()
                    k = 0
                    while(k < len(new_agg_values)):
                        if(new_agg_values[k] == new_topk_agg_value):
                            new_topk_indices.add(k)
                        k += 1
                    if(not original_topk_indices.issubset(new_topk_indices)):
                        actual_causes.add((flat_tuples[i].tuple, 0.5,))
                j += 1
            i += 1

        tuples_rho_geq_0_5 = set()
        tuples_rho_geq_0_5 = tuples_rho_geq_0_5.union(counterfactual_causes)
        tuples_rho_geq_0_5 = tuples_rho_geq_0_5.union(actual_causes)
        return list(tuples_rho_geq_0_5)


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
        
        # Dictionary to track input to output mapping
        self.input_to_output_mapping = collections.defaultdict(list)

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

    how_identifier_to_tuple_map = {}

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
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov

        self.scan_pointer = 1

        self.relation_tag = relation_tag
        self.column_headers = []

        with open(self.filepath) as input_file:
            input_reader = csv.reader(input_file, delimiter=" ")
            self.column_headers = input_reader.__next__()[1:]

    # Returns next batch of tuples in given file (or None if file exhausted)
    def get_next(self):
        output_data = []

        with open(self.filepath) as input_file:
            for idx, row in enumerate(input_file):
                if idx in range(self.scan_pointer, self.scan_pointer + self.batch_size):
                    processed_ATuple = ATuple(tuple=tuple(map(int, row.split(" "))), metadata=None, operator=self)
                    output_data.append(processed_ATuple)
                    
                    if(self.track_prov):
                        self.input_to_output_mapping[processed_ATuple] = processed_ATuple
                    
                    if(self.relation_tag == "L"):
                        processed_ATuple.metadata = "f"+str(idx)
                    if(self.relation_tag == "R"):
                        processed_ATuple.metadata = "r"+str(idx)
                    Scan.how_identifier_to_tuple_map[processed_ATuple.metadata] = processed_ATuple
                
        self.scan_pointer += self.batch_size
        
        if(output_data != []):
            annotated_output = [self.column_headers, output_data]
        else:
            annotated_output = [self.column_headers, None]
        
        return annotated_output

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        temp_lineage = []
        for element in tuples:
            temp_lineage += [self.input_to_output_mapping[element]]
        return temp_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        file_name = self.filepath.split("/")[-1]

        temp_where_prov = []
        for element in tuples:
            line_number = int(element.metadata[1:])
            temp_where_prov += [(file_name, line_number, self.input_to_output_mapping[element].tuple, element.tuple[att_index],)]

        return temp_where_prov

    # Starts the process of reading tuples (only for push-based evaluation)
    def start(self):
        while(True):
            output_data = []
            with open(self.filepath) as input_file:
                for idx, row in enumerate(input_file):
                    if idx in range(self.scan_pointer, self.scan_pointer + self.batch_size):
                        processed_ATuple = ATuple(tuple=tuple(map(int, row.split(" "))), metadata=None, operator=self)
                        output_data.append(processed_ATuple)

                        if(self.track_prov):
                            self.input_to_output_mapping[processed_ATuple] = processed_ATuple

                        if(self.relation_tag == "L"):
                            processed_ATuple.metadata = "f"+str(idx)
                        if(self.relation_tag == "R"):
                            processed_ATuple.metadata = "r"+str(idx)
                        Scan.how_identifier_to_tuple_map[processed_ATuple.metadata] = processed_ATuple

            self.scan_pointer += self.batch_size

            annotated_data = [self.column_headers, output_data, self.relation_tag] # the list contains information about the column headers in all the relations and information about the source of a relation (Left / Right)

            if(output_data != []):
                self.outputs[0].apply(annotated_data)
            else:
                self.outputs[0].apply([self.column_headers, None])
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
        
        self.left_inputs = left_inputs
        self.right_inputs = right_inputs
        self.left_join_attribute = left_join_attribute
        self.right_join_attribute = right_join_attribute
        self.outputs = outputs
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov

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

                            if(self.propagate_prov):
                                atup_left.metadata = "("+atup_left.metadata
                    else:
                        self.column_headers += tuples[0]
                        break
            
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
                        processed_tuple = (element.tuple, atup_right.tuple,)
                        processed_ATuple = ATuple(tuple=processed_tuple, operator=self)
                        output_from_probing.append(processed_ATuple)
                        
                        if(self.track_prov):
                            self.input_to_output_mapping[processed_ATuple].append(element)
                            self.input_to_output_mapping[processed_ATuple].append(atup_right)
                        
                        if(self.propagate_prov):
                            processed_ATuple.metadata = element.metadata+"*"+atup_right.metadata+")"

        if(tuples[1] is None):
            annotated_output = [self.column_headers, None]
        else:
            annotated_output = [self.column_headers, output_from_probing]

        return annotated_output

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        temp_lineage = []
        for element in tuples:
            temp_lineage += self.input_to_output_mapping[element][0].lineage()
            temp_lineage += self.input_to_output_mapping[element][1].lineage()
        return temp_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        temp_where_prov = []
        att_index = self.column_headers.index(att_index)

        for element in tuples:
            tuple_1_size = len(self.input_to_output_mapping[element][0].tuple)
            if(att_index < tuple_1_size):
                temp_where_prov += self.input_to_output_mapping[element][0].where(att_index)
                att_index = tuple_1_size
            else:
                att_index -= tuple_1_size
                temp_where_prov += self.input_to_output_mapping[element][1].where(att_index)
                
        return temp_where_prov

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        output_from_probing = []

        if(tuples[1] is not None):
            # split tuples L and R
            if(tuples[2] == "L"): # Probe the right relation hash, and then hash the left relation
                if(self.is_first_call_left):
                    self.column_headers += tuples[0]
                    self.is_first_call_left = False
                for tup in tuples[1]:
                    probing_key = str(tup.tuple[self.left_join_attribute])
                    if(self.right_relation_hash.get(probing_key) is not None):
                        matching_key = self.right_relation_hash[probing_key]
                        if(matching_key != []):
                            for element in matching_key:
                                processed_tuple = (element.tuple, tup.tuple,)
                                processed_ATuple = ATuple(tuple=processed_tuple, operator=self)
                                output_from_probing.append(processed_ATuple)

                                if(self.track_prov):
                                    self.input_to_output_mapping[processed_ATuple].append(element)
                                    self.input_to_output_mapping[processed_ATuple].append(tup)
                                
                                if(self.propagate_prov):
                                    processed_ATuple.metadata = "("+element.metadata +"*"+ tup.metadata+")"
    
                    self.left_relation_hash[str(tup.tuple[self.left_join_attribute])].append(tup)

            if(tuples[2] == "R"):
                if(self.is_first_call_right):
                    self.column_headers += tuples[0]
                    self.is_first_call_right = False
                for tup in tuples[1]:
                    probing_key = str(tup.tuple[self.right_join_attribute])
                    if(self.left_relation_hash.get(probing_key) is not None):
                        matching_key = self.left_relation_hash[probing_key]
                        if(matching_key != []):
                            for element in matching_key:
                                processed_tuple = (element.tuple, tup.tuple,)
                                processed_ATuple = ATuple(tuple=processed_tuple, operator=self)
                                output_from_probing.append(processed_ATuple)

                                if(self.track_prov):
                                    self.input_to_output_mapping[processed_ATuple].append(element)
                                    self.input_to_output_mapping[processed_ATuple].append(tup)

                                if(self.propagate_prov):
                                    processed_ATuple.metadata = "("+element.metadata +"*"+ tup.metadata+")"
                                
                    self.right_relation_hash[str(tup.tuple[self.right_join_attribute])].append(tup)

            annotated_output = [self.column_headers, output_from_probing]
            self.outputs[0].apply(annotated_output)
            return
        else:
            self.outputs[0].apply([tuples[0], None])



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

        self.inputs = inputs
        self.outputs = outputs
        self.fields_to_keep = fields_to_keep
        self.aliasing = aliasing
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov

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
                            for item in element.tuple:
                                for sub_item in item:
                                    flattened_tuple += (sub_item,)

                            temp = []
                            for index in project_fields_to_index:
                                temp.append(flattened_tuple[index])
                            processed_ATuple = ATuple(tuple=tuple(temp,), operator=self)
                            projected_column.append(processed_ATuple)

                            if(self.track_prov):
                                self.input_to_output_mapping[processed_ATuple] = element
                            
                            if(self.propagate_prov):
                                processed_ATuple.metadata = element.metadata

                    except: # For flat tuples
                        for element in tuples[1]: 
                            temp = []
                            for index in project_fields_to_index:
                                temp.append(element.tuple[index])
                            processed_ATuple = ATuple(tuple=tuple(temp,), operator=self)
                            projected_column.append(processed_ATuple)

                            if(self.track_prov):
                                self.input_to_output_mapping[processed_ATuple] = element
                            
                            if(self.propagate_prov):
                                processed_ATuple.metadata = element.metadata
        
        if(tuples[1] is None): 
            return [self.fields_to_keep, None]
        else:
            return [self.fields_to_keep, projected_column]

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        temp_lineage = []
        for element in tuples:
            temp_lineage += self.input_to_output_mapping[element].lineage()
        return temp_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        temp_lineage = []
        for element in tuples:
            temp_lineage += self.input_to_output_mapping[element].where(self.fields_to_keep[att_index])
        return temp_lineage

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        if(self.aliasing):
            annotated_output = [self.fields_to_keep, tuples[1]]
            self.outputs[0].apply(annotated_output)
        else:
            if(tuples[1] is not None):
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
                            for item in element.tuple:
                                for sub_item in item:
                                    flattened_tuple += (sub_item,)
                            temp = []
                            for index in project_fields_to_index:
                                temp.append(flattened_tuple[index])
                            processed_ATuple = ATuple(tuple=tuple(temp,), operator=self)
                            projected_column.append(processed_ATuple)

                            if(self.track_prov):    
                                self.input_to_output_mapping[processed_ATuple] = element
                            
                            if(self.propagate_prov):
                                processed_ATuple.metadata = element.metadata

                    except: # For flat tuples
                        for element in tuples[1]:
                            temp = []
                            for index in project_fields_to_index:
                                temp.append(element.tuple[index])
                            processed_ATuple = ATuple(tuple=tuple(temp,), operator=self)
                            projected_column.append(processed_ATuple)

                            if(self.track_prov):
                                self.input_to_output_mapping[processed_ATuple] = element
                            
                            if(self.propagate_prov):
                                processed_ATuple.metadata = element.metadata
                            
                    annotated_output = [self.fields_to_keep, projected_column]
                    self.outputs[0].apply(annotated_output)
                else:
                    return
            else:
                self.outputs[0].apply([tuples[0], None])



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

    how_provenance_dict = {}

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
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov

        self.column_headers = []
        self.grouping_dict = collections.defaultdict(list)
        self.intermediate_mapping = collections.defaultdict(list)
        self.metadata_dict = collections.defaultdict(str)
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

                        if(self.track_prov):
                            self.intermediate_mapping[str(self.key)] += [item]
                    else:
                        self.grouping_dict[str(item.tuple[self.key])].append(int(item.tuple[self.value]))

                        if(self.track_prov):
                            self.intermediate_mapping[str(item.tuple[self.key])].append(item)
                        
                        if(self.propagate_prov):
                            self.metadata_dict[str(item.tuple[self.key])] += item.metadata[:-1] + "@" + str((item.tuple[self.value])) +"), "

            if(self.key == 0 and self.value == 0):
                if(tuples[1] != [None]):
                    processed_ATuple = ATuple(
                                tuple=(self.agg_fun(self.grouping_dict[str(self.key)]),),
                                operator=self)
                    output_data += [processed_ATuple]
                
                if(self.track_prov):                  
                    self.input_to_output_mapping[processed_ATuple]  += self.intermediate_mapping[str(self.key)]

            else:
                for dict_item in enumerate(self.grouping_dict.items()):
                    output_dict[dict_item[1][self.key]] = self.agg_fun(dict_item[1][self.value])
                    processed_ATuple = ATuple(tuple=(int(dict_item[1][self.key]), output_dict[dict_item[1][self.key]]), operator=self)
                    output_data.append(processed_ATuple)

                    if(self.track_prov):
                        self.input_to_output_mapping[processed_ATuple] += self.intermediate_mapping[str(dict_item[1][self.key])]
                    
                    if(self.propagate_prov):
                        processed_ATuple.metadata = "AVG( "+self.metadata_dict[dict_item[1][self.key]][:-2]+" )"
        
                        GroupBy.how_provenance_dict[processed_ATuple.tuple] = processed_ATuple.metadata

        annotated_output = [tuples[0], output_data]

        if(tuples[1] is None): 
            return annotated_output
            
    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        temp_lineage = []
        for element in tuples:
            element_iter = self.input_to_output_mapping[element]
            for subelement in element_iter:
                temp_lineage += subelement.lineage()
        return temp_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        temp_where_prov = []
        for element in tuples:
            element_iter = self.input_to_output_mapping[element]
            for subelement in element_iter:
                temp_where_prov += subelement.where(att_index)

        return temp_where_prov

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        output_data = []
        if(tuples[1] is not None):
            self.column_headers = tuples[0]
            if(tuples[1] != []):
                for item in tuples[1]:
                    if(self.key == 0 and self.value == 0):
                        self.grouping_dict[str(self.key)].append(int(item.tuple[self.value]))

                        if(self.track_prov):
                            self.intermediate_mapping[str(self.key)] += [item]
                    
                    else:
                        self.grouping_dict[str(item.tuple[self.key])].append(int(item.tuple[self.value]))

                        if(self.track_prov):
                            self.intermediate_mapping[str(item.tuple[self.key])].append(item)
                        
                        if(self.propagate_prov):
                            self.metadata_dict[str(item.tuple[self.key])] += item.metadata[:-1] + "@" + str((item.tuple[self.value])) +"), "

            else:
                return
        else:
            if(self.is_first_none): # is nth none
                self.is_first_none = False
                return
            else:
                output_dict = {}
                if(self.key == 0 and self.value == 0):
                    try:
                        processed_ATuple = ATuple(
                                (self.agg_fun(self.grouping_dict[str(self.key)]),),
                                operator=self
                                )
                        output_data += [processed_ATuple]

                        if(self.track_prov):                  
                            self.input_to_output_mapping[processed_ATuple]  += self.intermediate_mapping[str(self.key)]
                    except:
                        if(self.outputs != []):
                            self.outputs[0].apply([self.column_headers, None])
                else:
                    for dict_item in enumerate(self.grouping_dict.items()):
                        output_dict[dict_item[1][self.key]] = self.agg_fun(dict_item[1][self.value])
                        processed_ATuple = ATuple(
                            tuple=(int(dict_item[1][self.key]), output_dict[dict_item[1][self.key]]), 
                            operator=self
                            )
                        output_data.append(processed_ATuple)

                        if(self.track_prov):
                            self.input_to_output_mapping[processed_ATuple] += self.intermediate_mapping[str(dict_item[1][self.key])]
                        
                        if(self.propagate_prov):
                            processed_ATuple.metadata = "AVG( "+self.metadata_dict[dict_item[1][self.key]][:-2]+" )"

                            GroupBy.how_provenance_dict[processed_ATuple.tuple] = processed_ATuple.metadata

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
        if(tuples[1] is not None):
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
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov

        self.collected_data = []

    # Returns the sorted input (or None if done)
    def get_next(self):
        intermediate_mapping = collections.defaultdict(list)
        metadata_mapping = collections.defaultdict(str)
        for operator in self.inputs:
            if(operator.name in ["Scan"]):
                while(True):
                    tuples = operator.get_next()
                    if(tuples[1] is None):
                        break
                    self.collected_data += tuples[1]
            else:
                tuples = operator.get_next()
                self.collected_data = tuples[1]

            tuple_list = []
            for element in self.collected_data:
                temp_tuple = element.tuple
                tuple_list.append(temp_tuple)

                if(self.track_prov):
                    intermediate_mapping[temp_tuple].append(element)
                
                if(self.propagate_prov):
                    metadata_mapping[temp_tuple] = element.metadata

            if(self.ASC):
                reversed = False
            else:
                reversed = True
            # Sort list by the first element
            sorted_list = self.comparator(tuple_list, key=lambda x:x[1], reverse=reversed)
            output_data = []
            for element in sorted_list:
                processed_ATuple = ATuple(tuple=element, operator=self)
                output_data.append(processed_ATuple)

                if(self.track_prov):
                    self.input_to_output_mapping[processed_ATuple] = intermediate_mapping[element]
                
                if(self.propagate_prov):
                    processed_ATuple.metadata = metadata_mapping[element]
        
        annotated_output = [tuples[0], output_data]

        return annotated_output
            
    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        temp_lineage = []
        for element in tuples:
            element_iter = self.input_to_output_mapping[element]
            for subelement in element_iter:
                temp_lineage += subelement.lineage()
        return temp_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        intermediate_mapping = collections.defaultdict(list)
        metadata_mapping = collections.defaultdict(str)

        self.collected_data = tuples[1]

        tuple_list = []
        for element in self.collected_data:
            temp_tuple = element.tuple
            tuple_list.append(temp_tuple)

            if(self.track_prov):
                intermediate_mapping[temp_tuple].append(element)
            
            if(self.propagate_prov):
                metadata_mapping[temp_tuple] = element.metadata
            

        if(self.ASC):
            reversed = False
        else:
            reversed = True
        # Sort list by the second element
        sorted_list = self.comparator(tuple_list, key=lambda x:x[1], reverse=reversed)
        output_data = []
        for element in sorted_list:
            processed_ATuple = ATuple(tuple=element, operator=self)
            output_data.append(processed_ATuple)

            if(self.track_prov):
                self.input_to_output_mapping[processed_ATuple] = intermediate_mapping[element]
            
            if(self.propagate_prov):
                processed_ATuple.metadata = metadata_mapping[element]

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
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov

        self.output_data = []

    # Returns the first k tuples in the input (or None if done)
    def get_next(self):
        for operator in self.inputs:
            if(operator.name in ["Scan", "Select", "Project"]):
                tuples = operator.get_next()
                while(self.k>=0):
                    if(tuples[1] is None):
                        break
                    self.output_data += tuples[1]
                    self.k -= len(self.output_data)
            else:
                tuples = operator.get_next()
                try:
                    self.output_data = tuples[1][0:self.k]
                except:
                    self.output_data = tuples[1]
                
                if(self.track_prov):
                    for element in self.output_data:
                        self.input_to_output_mapping[element] = element

            annotated_output = [tuples[0], self.output_data]

            return annotated_output


    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        temp_lineage = []
        for element in tuples:
            temp_lineage += self.input_to_output_mapping[element].lineage()
        return temp_lineage

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        # YOUR CODE HERE (ONLY FOR TASK 2 IN ASSIGNMENT 2)
        pass

    # Applies the operator logic to the given list of tuples
    # (TODO) Make TopK non-blocking by using k in while loop.
    def apply(self, tuples: List[ATuple]): 
        if(self.outputs != []):
            self.output_data = tuples[1][0:self.k]
            if(self.track_prov):
                for element in self.output_data:
                    self.input_to_output_mapping[element] = element

            annotated_output = [tuples[0], self.output_data]
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
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov

    # Returns next batch of tuples that pass the filter (or None if done)
    def get_next(self):
        processed_tuples = []
        for operator in self.inputs:
            tuples = operator.get_next()
            if(self.predicate is None):
                return tuples
            filtered_data = self.predicate(tuples[1])
            if(filtered_data != None and filtered_data != []):
                for element in filtered_data:
                    processed_tuples.append(element)
        if(tuples[1] is None):
            return [tuples[0], None]
        else:
            return [tuples[0], processed_tuples]

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        processed_tuples = []
        if(self.predicate is None):
            return tuples
        if(tuples[1] is not None):
            filtered_data = self.predicate(tuples[1])
            if(filtered_data != None and filtered_data != []):
                for element in filtered_data:
                    processed_tuples.append(element)
            self.outputs[0].apply([tuples[0], processed_tuples, tuples[2]])
            return
        else:
            self.outputs[0].apply([tuples[0], None])



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
        self.filepath = filepath
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov

        self.output_data = [] # stores the final output of the query

    # Returns next batch of tuples that pass the filter (or None if done)
    def get_next(self):
        for operator in self.inputs:
            tuples = operator.get_next()
            self.output_data = tuples[1]

            annotated_output = [tuples[0], self.output_data]

            if(not self.track_prov and not self.propagate_prov):
                self.write_to_csv(annotated_output)
            
    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        if(tuples[1] != [] and tuples[1] is not None):
            self.output_data += tuples[1]

        annotated_output = [tuples[0], self.output_data]

        if(not self.track_prov and not self.propagate_prov):
            self.write_to_csv(annotated_output)

    # Writes the tuples to the CSV file with column headers
    def write_to_csv(self, tuples):
        with open(self.filepath, 'w') as output_file:
            output_writer = csv.writer(output_file, delimiter=",")
            output_writer.writerow(tuples[0]) # Write column headers
            for item in tuples[1]:
                output_writer.writerow(item.tuple)
    
    def write_to_txt(self, content):       
        if(self.filepath != ""):
            with open(self.filepath, 'w') as output_file:
                output_file.write(str(content))

    # Collects and writes the lineage of the specified tuples
    def get_lineage(self, tuple_index):
        temp = self.output_data[tuple_index].lineage()
        write_to_file = []
        for item in temp:
            write_to_file.append(item.tuple)
            self.write_to_txt(write_to_file)
        return temp
    
    # Collects and writes the where provenance of the specified tuples
    def get_where_provenance(self, where_row_index, where_attribute_index):
        temp = self.output_data[where_row_index].where(where_attribute_index)
        write_to_file = []
        for item in temp:
            write_to_file.append(item)
            self.write_to_txt(write_to_file)
        return temp

    # Collects and writes the how provenance of the specified tuples
    def get_how_provenance(self, tuple_index):
        write_to_file = self.output_data[tuple_index].how()
        self.write_to_txt(write_to_file)
        return write_to_file
    
    # Collects and writes the responsibilities of the specified tuples
    def get_responsibility(self, tuple_index):
        self.get_how_provenance(tuple_index)
        write_to_file = self.output_data[tuple_index].responsible_inputs()
        self.write_to_txt(write_to_file)
        return write_to_file
        

## Filter functions to be passed to the "Select" operator           
def relation_filter_left(scan_output_left):
    if(scan_output_left is not None):
        filter_output_left = [value for value in scan_output_left if(value.tuple[0] == user_id)]
        return filter_output_left

def relation_filter_right(scan_output_right):
    if(scan_output_right is not None):
        filter_output_right = [value for value in scan_output_right if(value.tuple[1] == movie_id)]
        return filter_output_right

def pull_rating(relation_1, relation_2, relation_filter_left, relation_filter_right, output_path, track_prov, where_row_index=-1, where_attribute_index=-1):
    scan_operator_left = Scan(filepath=relation_1, outputs=[], relation_tag="L", track_prov=track_prov)
    scan_operator_right = Scan(filepath=relation_2, outputs=[], relation_tag="R", track_prov=track_prov)

    filter_operator_left = Select(inputs=[scan_operator_left], outputs=[], predicate=relation_filter_left, track_prov=track_prov)
    filter_operator_right = Select(inputs=[scan_operator_right], outputs=[], predicate=relation_filter_right, track_prov=track_prov)

    join_operator = Join(left_inputs=[filter_operator_left], right_inputs=[filter_operator_right], outputs=[], left_join_attribute=1, right_join_attribute=0, track_prov=track_prov)

    project_operator = Project(inputs=[join_operator], outputs=[], fields_to_keep=["Rating"], track_prov=track_prov)

    average_operator = GroupBy(inputs=[project_operator], outputs=[], agg_fun=mean, key=0, value=0, track_prov=track_prov)

    sink_operator = Sink(inputs=[average_operator], filepath=output_path, track_prov=track_prov)
    sink_operator.get_next()

    temp = None

    if(where_row_index != -1 and where_attribute_index != -1):
        temp = sink_operator.get_where_provenance(where_row_index, where_attribute_index)
    
    return temp

def push_rating(relation_1, relation_2, relation_filter_left, relation_filter_right, output_path, track_prov, where_row_index=-1, where_attribute_index=-1):
    sink_operator = Sink(inputs=[], filepath=output_path, track_prov=track_prov)

    average_operator = GroupBy(inputs=[], outputs=[sink_operator], agg_fun=mean, key=0, value=0, track_prov=track_prov) # key and value are attribute numbers after projection

    project_operator = Project(inputs=[], outputs=[average_operator], fields_to_keep=["Rating"], track_prov=track_prov)

    join_operator = Join(left_inputs=[], right_inputs=[], outputs=[project_operator], left_join_attribute=1, right_join_attribute=0, track_prov=track_prov)

    filter_operator_left = Select(inputs=[], outputs=[join_operator], predicate=relation_filter_left, track_prov=track_prov)
    filter_operator_right = Select(inputs=[], outputs=[join_operator], predicate=relation_filter_right, track_prov=track_prov)  

    scan_operator_left = Scan(filepath=relation_1, outputs=[filter_operator_left], relation_tag="L", track_prov=track_prov)
    scan_operator_right = Scan(filepath=relation_2, outputs=[filter_operator_right], relation_tag="R", track_prov=track_prov)

    scan_operator_left.start()
    scan_operator_right.start()

    temp = None

    if(where_row_index != -1 and where_attribute_index != -1):
        temp = sink_operator.get_where_provenance(where_row_index, where_attribute_index)
        
    return temp



def pull_recommendation(relation_1, relation_2, relation_filter, output_path, track_prov, propagate_prov, lineage_tuple_index, how_tuple_index, responsibility_tuple_index):

    scan_operator_left = Scan(filepath=relation_1, outputs=[], relation_tag="L", track_prov=track_prov, propagate_prov = propagate_prov)
    scan_operator_right = Scan(filepath=relation_2, outputs=[], relation_tag="R", track_prov=track_prov, propagate_prov = propagate_prov)

    filter_operator_left = Select(inputs=[scan_operator_left], outputs=[], predicate=relation_filter, track_prov=track_prov, propagate_prov = propagate_prov)

    join_operator = Join(left_inputs=[filter_operator_left], right_inputs=[scan_operator_right], outputs=[], left_join_attribute=1, right_join_attribute=0, track_prov=track_prov, propagate_prov = propagate_prov)

    project_operator = Project(inputs=[join_operator], outputs=[], fields_to_keep=["MID", "Rating"], track_prov=track_prov, propagate_prov = propagate_prov)

    groupby_operator = GroupBy(inputs=[project_operator], outputs=[], agg_fun=mean, key=0, value=1, track_prov=track_prov, propagate_prov = propagate_prov)

    orderby_operator = OrderBy(inputs=[groupby_operator], outputs=[], comparator=sorted, ASC=False, track_prov=track_prov, propagate_prov = propagate_prov)

    limit_operator = TopK(inputs=[orderby_operator], outputs=[], k=1, track_prov=track_prov, propagate_prov = propagate_prov)

    project_operator = Project(inputs=[limit_operator], outputs=[], fields_to_keep=["MID"], aliasing=False, track_prov=track_prov, propagate_prov = propagate_prov)

    sink_operator = Sink(inputs=[project_operator], filepath=output_path, track_prov=track_prov, propagate_prov=propagate_prov)
    sink_operator.get_next()

    temp = None

    if(track_prov):
        temp = sink_operator.get_lineage(lineage_tuple_index)

    if(propagate_prov):
        temp = sink_operator.get_how_provenance(how_tuple_index)
    
    if(responsibility_tuple_index != -1):
        temp = sink_operator.get_responsibility(responsibility_tuple_index)
    
    return temp

def push_recommendation(relation_1, relation_2, relation_filter, output_path, track_prov, propagate_prov, lineage_tuple_index, how_tuple_index, responsibility_tuple_index):

    sink_operator = Sink(inputs=[], filepath=output_path, track_prov=track_prov, propagate_prov=propagate_prov)

    project_operator = Project(inputs=[], outputs=[sink_operator], fields_to_keep=["MID"], aliasing=False, track_prov=track_prov, propagate_prov=propagate_prov)

    limit_operator = TopK(inputs=[], outputs=[project_operator], k=1, track_prov=track_prov, propagate_prov=propagate_prov)

    orderby_operator = OrderBy(inputs=[], outputs=[limit_operator], comparator=sorted, ASC=False, track_prov=track_prov, propagate_prov=propagate_prov)

    groupby_operator = GroupBy(inputs=[], outputs=[orderby_operator], agg_fun=mean, key=0, value=1, track_prov=track_prov, propagate_prov=propagate_prov) # key and value are attribute numbers after projection

    project_operator = Project(inputs=[], outputs=[groupby_operator], fields_to_keep=["MID", "Rating"], track_prov=track_prov, propagate_prov=propagate_prov)

    join_operator = Join(left_inputs=[], right_inputs=[], outputs=[project_operator], left_join_attribute=1, right_join_attribute=0, track_prov=track_prov, propagate_prov=propagate_prov)

    filter_operator_left = Select(inputs=[], outputs=[join_operator], predicate=relation_filter, track_prov=track_prov, propagate_prov=propagate_prov) 

    scan_operator_left = Scan(filepath=relation_1, outputs=[filter_operator_left], relation_tag="L", track_prov=track_prov, propagate_prov=propagate_prov)
    scan_operator_right = Scan(filepath=relation_2, outputs=[join_operator], relation_tag="R", track_prov=track_prov, propagate_prov=propagate_prov)

    scan_operator_left.start()
    scan_operator_right.start()
    
    temp = None

    if(track_prov):
        temp = sink_operator.get_lineage(lineage_tuple_index)

    if(propagate_prov):
        temp = sink_operator.get_how_provenance(how_tuple_index)
    
    if(responsibility_tuple_index != -1):
        temp = sink_operator.get_responsibility(responsibility_tuple_index)
    
    return temp
        

## Driver
if __name__ == "__main__":

    logger.info("Assignment #1")

    # Initialize parser
    parser = argparse.ArgumentParser()

    options_list = ["query", "ff", "mf", "uid", "mid", "pull", "output", "lineage", "where-row", "where-attribute", "how", "responsibility"]

    for argument in options_list:
        if(argument in ["lineage", "where-row", "where-attribute", "how", "responsibility"]):
            parser.add_argument("--"+argument, nargs='?', default=-1)
        else:
            parser.add_argument("--"+argument)

    args = parser.parse_args()

    movie_id = int(args.mid)
    user_id = int(args.uid)

    relation_1 = args.ff
    relation_2 = args.mf

    # --lineage [int] --where-row [int] --where-attribute [int] --how [int] --responsibility [int]

    track_prov = False
    if(args.lineage == -1):
        track_prov = False
    else:
        track_prov = True
    lineage_tuple_index = int(args.lineage)
    
    if(args.where_row != -1 and args.where_attribute):
        track_prov = True
    where_row_index = int(args.where_row)  
    where_attribute_index = int(args.where_attribute)
    
    propagate_prov = False
    if(args.how == -1):
        propagate_prov = False
    else:
        propagate_prov = True
    how_tuple_index = int(args.how)
    
    if(args.responsibility == -1):
        responsibility_tuple_index = -1
    else:
        propagate_prov = True
    how_tuple_index = int(args.responsibility)
    responsibility_tuple_index = int(args.responsibility)

    # TASK 1: Implement 'likeness' prediction query for User A and Movie M
    #
    # SELECT AVG(R.Rating)
    # FROM Friends as F, Ratings as R
    # WHERE F.UID2 = R.UID
    #       AND F.UID1 = 'A'
    #       AND R.MID = 'M'

    if(args.query == "1"):
        if(args.pull == "1"):
            ## -------------------------
            ## Pull-based
            ## -------------------------
            pull_rating(relation_1=relation_1, relation_2=relation_2, relation_filter_left=relation_filter_left, relation_filter_right=relation_filter_right, output_path=args.output, track_prov=track_prov, where_row_index=where_row_index, where_attribute_index=where_attribute_index)
                    
        else:
            ## -------------------------
            ## Push-based
            ## -------------------------
            push_rating(relation_1=relation_1, relation_2=relation_2, relation_filter_left=relation_filter_left, relation_filter_right=relation_filter_right, output_path=args.output, track_prov=track_prov, where_row_index=where_row_index, where_attribute_index=where_attribute_index)
    

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

    if(args.query == "2"):
        if(args.pull == "1"):
            ## -------------------------
            ## Pull-based
            ## -------------------------
            pull_recommendation(relation_1=relation_1, relation_2=relation_2, relation_filter=relation_filter_left, output_path=args.output, track_prov=track_prov, propagate_prov=propagate_prov, lineage_tuple_index=lineage_tuple_index,how_tuple_index=how_tuple_index, responsibility_tuple_index=responsibility_tuple_index)
        else:
            ## -------------------------
            ## Push-based
            ## -------------------------
            push_recommendation(relation_1=relation_1, relation_2=relation_2, relation_filter=relation_filter_left, output_path=args.output, track_prov=track_prov, propagate_prov=propagate_prov, lineage_tuple_index=lineage_tuple_index, how_tuple_index=how_tuple_index, responsibility_tuple_index=responsibility_tuple_index)

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

            sink_operator = Sink(inputs=[project_operator], filepath=args.output)
            sink_operator.get_next()
        else:
            ## -------------------------
            ## Push-based
            ## -------------------------
            sink_operator = Sink(inputs=[], filepath=args.output)

            project_operator = Project(inputs=[], outputs=[sink_operator], fields_to_keep=["Rating", "Count"], aliasing=True)

            histogram_operator = Histogram(inputs=[], outputs=[project_operator], key=0) # key is the attribute index after projection

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

    # The TASK is decided based on the arguments passed in the execution command in the CLI

    # TASK 2: Implement where-provenance query for 'likeness' prediction

    if(args.query == "1"):
        if(args.pull == "1"):
            ## -------------------------
            ## Pull-based
            ## -------------------------
            pull_rating(relation_1=relation_1, relation_2=relation_2, relation_filter_left=relation_filter_left, relation_filter_right=relation_filter_right, output_path=args.output, track_prov=track_prov, where_row_index=where_row_index, where_attribute_index=where_attribute_index)
                    
        else:
            ## -------------------------
            ## Push-based
            ## -------------------------
            push_rating(relation_1=relation_1, relation_2=relation_2, relation_filter_left=relation_filter_left, relation_filter_right=relation_filter_right, output_path=args.output, track_prov=track_prov, where_row_index=where_row_index, where_attribute_index=where_attribute_index)

    
    # TASK 1: Implement lineage query for movie recommendation
    # TASK 3: Implement how-provenance query for movie recommendation
    # TASK 4: Retrieve most responsible tuples for movie recommendation

    if(args.query == "2"):
        if(args.pull == "1"):
            ## -------------------------
            ## Pull-based
            ## -------------------------
            pull_recommendation(relation_1=relation_1, relation_2=relation_2, relation_filter=relation_filter_left, output_path=args.output, track_prov=track_prov, propagate_prov=propagate_prov, lineage_tuple_index=lineage_tuple_index, how_tuple_index=how_tuple_index, responsibility_tuple_index=responsibility_tuple_index)
        else:
            ## -------------------------
            ## Push-based
            ## -------------------------
            push_recommendation(relation_1=relation_1, relation_2=relation_2, relation_filter=relation_filter_left, output_path=args.output, track_prov=track_prov, propagate_prov=propagate_prov, lineage_tuple_index=lineage_tuple_index, how_tuple_index=how_tuple_index, responsibility_tuple_index=responsibility_tuple_index)
