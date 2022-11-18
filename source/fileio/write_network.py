from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class WriteNetwork(ABC):
    """
    Abstract base class for the implementations related to writing the results to files.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of WriteNetwork.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def write(self, flownetwork):
        """
        Write the network and results to a file
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class WriteNetworkNothing(WriteNetwork):
    """
    Class for not writing anything
    """

    def write(self, flownetwork):
        """
        Do not write anything
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class WriteNetworkIgraph(WriteNetwork):
    """
    Class for writing the results to igraph format.
    """

    def write(self, flownetwork):
        """
        Write the network and simulation data into an igraph file
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        import igraph

        if self._PARAMETERS["write_override_initial_graph"]:
            graph = igraph.Graph(flownetwork.edge_list.tolist())
            # todo: how to handle overwrite?
            pass
        else:
            edge_list = flownetwork.edge_list
            graph = igraph.Graph(edge_list.tolist())  # Generate igraph based on edge_list

        graph.es["diameter"] = flownetwork.diameter
        graph.es["length"] = flownetwork.length
        graph.es["flow_rate"] = flownetwork.flow_rate
        graph.es["rbc_velocity"] = flownetwork.rbc_velocity
        graph.es["ht"] = flownetwork.ht

        graph.vs["xyz"] = flownetwork.xyz.tolist()
        graph.vs["pressure"] = flownetwork.pressure

        graph.write_pickle(self._PARAMETERS["write_path_igraph"])
        # todo: check that old graph is not overwritten
        # todo: handle boundaries