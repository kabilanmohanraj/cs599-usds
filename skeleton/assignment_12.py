from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
import collections

import csv
from email import header
import logging
from enum import Enum
from turtle import right
from typing import List, Tuple
import uuid
from numpy import average

from yaml import scan

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
                 batch_size=100,
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

        with open(self.filepath) as input_file:
            input_reader = csv.reader(input_file, delimiter=" ")
            column_headers = input_reader.__next__()[1:]
        
        for i in range(len(column_headers)): # Assign indices to the column headers
            Scan.column_headers_to_index[column_headers[i]] = str(Scan.scan_operator_number)+"_"+str(i)
        Scan.scan_operator_number += 1

    # Returns next batch of tuples in given file (or None if file exhausted)
    def get_next(self):
        logger.debug("In scan opertor =====")
        output_data = []

        with open(self.filepath) as input_file:
            output_data = [ATuple(tuple = tuple(map(int, row.split(" ")))) for idx, row in enumerate(input_file) if idx in range(self.scan_pointer, self.scan_pointer + self.batch_size)]

        self.scan_pointer += self.batch_size

        if(output_data != []):
            return output_data, Scan.column_headers_to_index
        else:
            return None, Scan.column_headers_to_index

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
        # while(True):
        logger.debug("In scan opertor push-based =====")
        output_data = []
        while(True):
            with open(self.filepath) as input_file:
                output_data = [ATuple(tuple = tuple(map(int, row.split(" ")))) for idx, row in enumerate(input_file) if idx in range(self.scan_pointer, self.scan_pointer + self.batch_size)]

            self.scan_pointer += self.batch_size

            annotated_data = [Scan.column_headers_to_index, output_data, self.relation_tag] # the list contains information about the column headers in all the relations and information about the source relation (Left / Right)

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

        self.is_first_call = True
        self.left_relation_hash = collections.defaultdict(list)
        self.right_relation_hash = collections.defaultdict(list)
        self.table_to_hash = 0

    # Returns next batch of joined tuples (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        logger.info("In Join operator pull-based======")
        print()

        output_from_probing = []

        if(self.is_first_call):
            self.is_first_call = False
            for operator in self.left_inputs:
                while(True):
                    left_input, column_headers_to_index = operator.get_next()
                    if(left_input is None):
                        break
                    # Collect the left relation into a dictionary (hashing) only on the first call
                    for atup_left in left_input:
                        self.left_relation_hash[str(atup_left.tuple[self.left_join_attribute])].append(atup_left)
            
            # logger.debug(self.left_relation_hash["1769"]) # returns empty list if key not found
            logger.info("Completed hashing phase. Starting probing phase====")

        for operator in self.right_inputs:
            # while(True):
            right_input, column_headers_to_index = operator.get_next()
            if(right_input is None):
                break
            # Collect the right relation in batches and compare with the previously created dictionary (probing)
            for atup_right in right_input:
                matching_key = self.left_relation_hash[str(atup_right.tuple[self.right_join_attribute])]
                if(matching_key != []):
                    for element in matching_key:
                        output_from_probing.append((element, atup_right))

        # logger.debug(output_from_probing[0])
        return output_from_probing, column_headers_to_index # list of tuple of tuples

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
            # if(self.table_to_hash%2):
            if(tuples[2] == "L"): # Probe the right relation hash and then hash the left relation
                for tup in tuples[1]:
                    probing_key = str(tup.tuple[self.left_join_attribute])
                    if(probing_key in self.right_relation_hash.keys()):
                        matching_key = self.right_relation_hash[probing_key]
                        if(matching_key != []):
                            for element in matching_key:
                                output_from_probing.append((element, tup))
                    self.left_relation_hash[str(tup.tuple[self.left_join_attribute])].append(tup)

            if(tuples[2] == "R"):
                for tup in tuples[1]:
                    probing_key = str(tup.tuple[self.right_join_attribute])
                    if(probing_key in self.left_relation_hash.keys()):
                        matching_key = self.left_relation_hash[probing_key]
                        if(matching_key != []):
                            for element in matching_key:
                                output_from_probing.append((element, tup))
                self.right_relation_hash[str(tup.tuple[self.right_join_attribute])].append(tup)

            annotated_output = [tuples[0], output_from_probing, tuples[2]]
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

        self.column_headers_to_index = []

    # Return next batch of projected tuples (or None if done)
    def get_next(self):
        projected_column = []
        for operator in self.inputs:
            output, column_headers_to_index = operator.get_next()
            self.column_headers_to_index = column_headers_to_index
            project_fields_to_index = list(map(int, column_headers_to_index[self.fields_to_keep[0]].split("_")))
            
            for element in output:
                projected_column.append(element[project_fields_to_index[0]].tuple[project_fields_to_index[1]])
        
        return projected_column, column_headers_to_index

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
        if(tuples is not None):
            projected_column = []
            column_headers_to_index = tuples[0]
            project_fields_to_index = list(map(int, column_headers_to_index[self.fields_to_keep[0]].split("_")))
            
            if(tuples[1] != []):
                for element in tuples[1]:
                    projected_column.append(element[project_fields_to_index[0]].tuple[project_fields_to_index[1]])
            
                self.outputs[0].apply(projected_column)
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

        self.input_tuples = []

    # Returns aggregated value per distinct key in the input (or None if done)
    def get_next(self):
        logger.debug("In group by operator===")
        for operator in self.inputs:
            while(True):
                output_from_probing, column_headers_to_index = operator.get_next()
                if(output_from_probing == []): # is None
                    break
                self.input_tuples += output_from_probing # this is a list of list
            logger.debug(self.agg_fun(self.input_tuples))

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
        if(tuples is not None):
            if(tuples != []):
                self.input_tuples += tuples
            else:
                return
        else:
            if(self.input_tuples != []):
                logger.debug(self.agg_fun(self.input_tuples))

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
        # YOUR CODE HERE
        pass

    # Returns histogram (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        pass

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        pass

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
        # YOUR CODE HERE
        pass

    # Returns the sorted input (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        pass

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
        pass

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
        # YOUR CODE HERE
        pass

    # Returns the first k tuples in the input (or None if done)
    def get_next(self):
        # YOUR CODE HERE
        pass

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
        pass

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
        logger.debug("In select operator===")
        for operator in self.inputs:
            output_data, column_headers_to_index = operator.get_next()
            filtered_output_data = self.predicate(output_data)
        return filtered_output_data, column_headers_to_index

    # Applies the operator logic to the given list of tuples
    def apply(self, tuples: List[ATuple]):
        if(tuples is not None):
            filtered_output_data = self.predicate(tuples[1])
            tuples[1] = filtered_output_data.copy()
            self.outputs[0].apply(tuples)
            return
        else:
            self.outputs[0].apply(None)
            

        

if __name__ == "__main__":

    logger.info("Assignment #1")

    # TASK 1: Implement 'likeness' prediction query for User A and Movie M
    #
    # SELECT AVG(R.Rating)
    # FROM Friends as F, Ratings as R
    # WHERE F.UID2 = R.UID
    #       AND F.UID1 = 'A'
    #       AND R.MID = 'M'

    # YOUR CODE HERE

    batch_size = int(input("Please enter preferred batch size for reading input relations: "))
    movie_id = int(input("Please enter the movie id: "))
    user_id = int(input("Please enter the user id: "))

    ## -------------------------
    ## Pull-based
    ## -------------------------

    relation_1 = "./data/friends.txt"
    relation_2 = "./data/movie_ratings.txt"
    # scan_operator_left = Scan(filepath=relation_1, outputs=[], batch_size=batch_size, relation_tag="L")
    # scan_operator_right = Scan(filepath=relation_2, outputs=[], batch_size=batch_size, relation_tag="R")


    def relation_filter_left(scan_output_left):
        if(scan_output_left is not None):
            filter_output_left = [row for row in scan_output_left if(row.tuple[0] == user_id)]
            return filter_output_left

    def relation_filter_right(scan_output_right):
        if(scan_output_right is not None):
            filter_output_right = [row for row in scan_output_right if(row.tuple[1] == movie_id)]
            return filter_output_right

    # filter_operator_left = Select(inputs=[scan_operator_left], outputs=[], predicate=relation_filter_left)
    # filter_operator_right = Select(inputs=[scan_operator_right], outputs=[], predicate=relation_filter_right)


    # join_operator = Join(left_inputs=[filter_operator_left], right_inputs=[filter_operator_right], outputs=[], left_join_attribute=1, right_join_attribute=0)


    # project_operator = Project(inputs=[join_operator], outputs=[], fields_to_keep=["Rating"])


    def avg(input_tuples):
        return average(input_tuples)

    # average_operator = GroupBy(inputs=[project_operator], outputs=[], agg_fun=avg, key=4 , value=5)
    # average_operator.get_next()



    ## -------------------------
    ## Push-based
    ## -------------------------

    average_operator = GroupBy(inputs=[], outputs=[], agg_fun=avg, key=4 , value=5)

    project_operator = Project(inputs=[], outputs=[average_operator], fields_to_keep=["Rating"])

    join_operator = Join(left_inputs=[], right_inputs=[], outputs=[project_operator], left_join_attribute=1, right_join_attribute=0)

    filter_operator_left = Select(inputs=[], outputs=[join_operator], predicate=relation_filter_left)
    filter_operator_right = Select(inputs=[], outputs=[join_operator], predicate=relation_filter_right)    

    scan_operator_left = Scan(filepath=relation_1, outputs=[filter_operator_left], batch_size=batch_size, relation_tag="L")
    scan_operator_right = Scan(filepath=relation_2, outputs=[filter_operator_right], batch_size=batch_size, relation_tag="R")

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


    # TASK 3: Implement explanation query for User A and Movie M
    #
    # SELECT HIST(R.Rating) as explanation
    # FROM Friends as F, Ratings as R
    # WHERE F.UID2 = R.UID
    #       AND F.UID1 = 'A'
    #       AND R.MID = 'M'

    # YOUR CODE HERE


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
