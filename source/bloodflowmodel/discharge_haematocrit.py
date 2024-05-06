from abc import ABC, abstractmethod
from types import MappingProxyType
import sys
import numpy as np


class DischargeHaematocrit(ABC):
    """
    Abstract base class for the implementations related to calculating the discharge haematocrit.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of DischargeHaematocrit
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def update_hd(self, flownetwork):
        """
        Abstract method to update the discharge haematocrit in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class DischargeHaematocritNewtonian(DischargeHaematocrit):
    """
    Class for updating the discharge haematocrit without taking red blood cells into account.
    """

    def update_hd(self, flownetwork):
        """
        Update the discharge haematocrit in the flow network with the zero vector for Newtonian flow.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.hd = flownetwork.ht


class DischargeHaematocritVitroPries1992(DischargeHaematocrit):
    """
    Class for updating the discharge haematocrit based on the empirical in vitro functions by
    Pries, Neuhaus, Gaehtgens (1992).
    """

    def update_hd(self, flownetwork):
        """
        Update the discharge haematocrit in flownetwork based on tube haematocrit and vessel diameter. The
        model is based on the empirical in vitro functions by Pries, Neuhaus, Gaehtgens (1992).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        ht = flownetwork.ht  # Tube haematocrit
        diameter_um = 1.e6 * flownetwork.diameter  # Vessel diameter in micro meters

        # Correct the diameters to compute the Hd
        diameter_um_correct = np.copy(diameter_um)
        diameter_um_correct[diameter_um < 3.] = 3.  # Vessel diameter in micro meters

        x_tmp = 1. + 1.7 * np.exp(-0.35*diameter_um_correct) - 0.6 * np.exp(-0.01*diameter_um_correct)  # Eq. (9) in paper
        x_bound = np.copy(x_tmp)
        x_bound[x_tmp > 0.9999] = 0.9999  # Bound x to values < 1. Equation in paper is only valid for x < 1.

        hd = -x_bound / (2 - 2*x_bound) + np.sqrt(
            np.square(x_bound / (2 - 2*x_bound)) + ht / (1-x_bound))  # Eq 10 in paper
        hd[x_tmp > 0.9999] = ht[x_tmp > 0.9999]  # For very small and very large diameters: set ht=hd

        flownetwork.hd = hd  # Update discharge haematocrit


class DischargeHaematocritVitroPries2005(DischargeHaematocrit):
    """
    Class for updating the discharge haematocrit based on the empirical in vitro functions by
    Pries and Secomb (2005).
    """

    def update_hd(self, flownetwork):
        """
        Update the discharge haematocrit in flownetwork based on tube haematocrit and vessel diameter. The
        model is based on the empirical in vitro functions by Pries and Secomb (2005).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        ht = flownetwork.ht  # Tube haematocrit
        diameter_um = 1.e6 * flownetwork.diameter  # Vessel diameter in micro meters

        # Correct the diameters to compute the Hd
        diameter_um_correct = np.copy(diameter_um)
        diameter_um_correct[diameter_um < 3.] = 3.  # Vessel diameter in micro meters

        x_tmp = 1. + 1.7 * np.exp(-0.415 * diameter_um_correct) - 0.6 * np.exp(-0.011 * diameter_um_correct) # From Eq.(1) in paper
        x_bound = np.copy(x_tmp)
        x_bound[x_tmp > 0.9999] = 0.9999  # Bound x to values < 1. Equation in paper is only valid for x < 1.

        hd = -x_bound / (2 - 2*x_bound) + np.sqrt(
            np.square(x_bound / (2 - 2*x_bound)) + ht / (1-x_bound))
        hd[x_tmp > 0.9999] = ht[x_tmp > 0.9999]  # For very small and very large diameters: set ht=hd

        flownetwork.hd = hd  # Update discharge haematocrit

class DischargeHaematocritPayne2023(DischargeHaematocrit):
    """
    Class for setting a constant value for the discharge haematocrit suggested by Payne et al., (2023).
    """

    def update_hd(self, flownetwork):
        """
        Set a constant value for the discharge haematocrit suggested by Payne et al. (2023)
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        hd_constant = 0.42  # For simplicity, hd is assumed here to have a uniform value of 0.42

        flownetwork.hd = np.ones(flownetwork.nr_of_es, dtype=np.float) * hd_constant


