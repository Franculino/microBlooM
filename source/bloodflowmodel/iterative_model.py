"""

To calculate the different hematocrit in the graph is needed to be calculated for each vessel

for this reason it has been added an iflow vessel and out flow vessul to the fra ph

"""

from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np


class Iterative(ABC):
    """
    Abstract base class for the implementations of iterative approach
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of Iterative.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def iterative_in_out(self, flownetwork):
        """
        checks if the network has an in and outflows
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def create_pressure_order(self, flownetwork):
        """
        create a dict that contain the ordered pressure and node
        to be albe to visit them in order from higher to lower pressure
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def to_store_hematocrit_values(self, flownetwork):
        """
        to store at each iteration the hematocrit
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


def first_node(flownetwork):
    # list of the boundary vertex
    boundary_vs = flownetwork.boundary_vs
    # list of the connection
    edge_list = flownetwork.edge_list
    # return the possible connection of the first node
    connection = []

    # for node in edge_list

    # if
    connection = [connection.append(node) if edge_list[node][0] == boundary_vs[0] else None for node in edge_list]
    return boundary_vs[0]


class IterativeModel(Iterative):

    def iterative_in_out(self, flownetwork):
        """
        Starting from the first node in the adjacent list defined as starting point and one in and one out:
        case 0: ( - ) structure.
                There is one vessel after the node, and it is the first
        case 1: ( < ) structure.
                There are two out vessel
                case 1.1: ( -< ) structure.
                    There are one in vessel in to a doble out vessel
        case 2: ( >- ) structure.
                There are two in vessel into a single out vessel

        """

        # get the first node
        first_node(flownetwork)
        # Case 0: (-) strucuture

        # Case 1: (<) strucuture

        # Case 1: (>-) strucuture

        pass
