from abc import ABC, abstractmethod
from types import MappingProxyType
import sys
import numpy as np


class TubeHaematocrit(ABC):
    """
    Abstract base class for the implementations related to calculating the tube haematocrit.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of TubeHaematocrit.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def update_ht(self, flownetwork):
        """
        Update the tube haematocrit in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class TubeHaematocritNewtonian(TubeHaematocrit):
    """
    Class for updating the tube haematocrit without red blood cells.
    """

    def update_ht(self, flownetwork):
        """
        Update the tube haematocrit in flownetwork with the zero vector without red blood cells.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.ht = np.zeros(flownetwork.nr_of_es, dtype=np.float)


class TubeHaematocritConstant(TubeHaematocrit):
    """
    Class for updating the tube haematocrit based on an enforced constant tube haematocrit.
    """

    def update_ht(self, flownetwork):
        """
        Update the tube haematocrit in flownetwork based on an enforced constant tube haematocrit.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        ht_constant = np.float(self._PARAMETERS["ht_constant"])
        if ht_constant < 0 or ht_constant > 1:
            sys.exit("Error: Ht has to be in range 0 to 1.")

        # Assign constant haematocrit to all edges
        flownetwork.ht = np.ones(flownetwork.nr_of_es, dtype=np.float) * ht_constant


class TubeHaematocritTracking(TubeHaematocrit):
    def update_ht(self, flownetwork):
        pass
