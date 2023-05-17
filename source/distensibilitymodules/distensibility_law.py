from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class DistensibilityLaw(ABC):
    """
    Abstract base class for bla
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of ReadTargetValues.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def initialise_distensibility_law(self, distensibility, flownetwork):
        """
        Read target values
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def update_diameter(self, distensibility, flownetwork):
        """
        Read target values
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class DistensibilityLawPassiveAnalytic(DistensibilityLaw):

    def initialise_distensibility_law(self, distensibility, flownetwork):
        """
        Read target values
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        distensibility.pressure_ref = np.copy(flownetwork.pressure)
        distensibility.diameter_ref = np.copy(flownetwork.diameter)

    def update_diameter(self, distensibility, flownetwork):
        """
        Read target values
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        nu = 0.5
        # compute mean pressure for each vessel based on pressures in adjacent vertices
        pressure_difference_vertex = flownetwork.pressure - distensibility.pressure_ref
        pressure_difference_edge = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)

        can_adapt = distensibility.can_adapt

        diameter_new = np.copy(flownetwork.diameter)

        diameter_new[can_adapt] = distensibility.diameter_ref[can_adapt] + pressure_difference_edge[can_adapt] * np.square(
            distensibility.diameter_ref[can_adapt]) * (1 - np.square(nu)) / (
                                   2. * distensibility.e_modulus[can_adapt] * distensibility.wall_thickness[can_adapt])

        flownetwork.diameter = diameter_new
