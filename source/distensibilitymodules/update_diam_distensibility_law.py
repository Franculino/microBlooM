from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class DistensibilityLawUpdate(ABC):
    """
    Abstract base class for updating the diameters
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of DistensibilityLawUpdate.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def update_diameter(self, distensibility, flownetwork):
        """
        Update the diameters based on a distensibility law
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class DistensibilityLawUpdateNothing(DistensibilityLawUpdate):

    def update_diameter(self, distensibility, flownetwork):
        """
        Do not update any diameters based on vessel distensibility
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class DistensibilityLawUpdatePassiveSherwin(DistensibilityLawUpdate):

    def update_diameter(self, distensibility, flownetwork):
        """
        Update the diameters based on a non-linear p-A ralation proposed by Sherwin et al. (2003)
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_dist = distensibility.eid_vessel_distensibility
        # Compute p-p_ref for each vertex of the entire network
        pressure_difference_vertex = flownetwork.pressure - distensibility.pressure_ref
        # Compute p-p_ref for each edge that can change due to the distensibility. Take the mean pressure of adjacent
        # vertices to get the edge-based pressure difference. Only compute for distensible edges.
        pressure_difference_dist_edges = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[
            eids_dist]

        diameter_new = np.copy(flownetwork.diameter)
        # Compute the updated diameters
        diameter_new[eids_dist] = distensibility.diameter_ref + pressure_difference_dist_edges * np.square(
            distensibility.diameter_ref) * (1 - np.square(distensibility.nu)) / (
                                              2. * distensibility.e_modulus * distensibility.wall_thickness)
        # Update diameters
        flownetwork.diameter = diameter_new


class DistensibilityLawUpdatePassiveUrquiza(DistensibilityLawUpdate):

    def update_diameter(self, distensibility, flownetwork):
        """
        Update the diameters based on a non-linear p-A relation proposed by Urquiza et al. (2006)
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_dist = distensibility.eid_vessel_distensibility
        # Compute p-p_ref for each vertex of the entire network
        pressure_difference_vertex = flownetwork.pressure - distensibility.pressure_ref
        # Compute p-p_ref for each edge that can change due to the distensibility. Take the mean pressure of adjacent
        # vertices to get the edge-based pressure difference. Only compute for distensible edges.
        pressure_difference_dist_edges = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[
            eids_dist]

        diameter_new = np.copy(flownetwork.diameter)
        # Compute the updated diameters
        diameter_new[eids_dist] = distensibility.diameter_ref + pressure_difference_dist_edges * np.square(
            distensibility.diameter_ref) / (2. * distensibility.e_modulus * distensibility.wall_thickness)
        # Update diameters
        flownetwork.diameter = diameter_new


class DistensibilityLawUpdatePassiveRammos(DistensibilityLawUpdate):

    def update_diameter(self, distensibility, flownetwork):
        """
        Update the diameters based on a linear p-A relation proposed by Rammos et al. (1998)
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_dist = distensibility.eid_vessel_distensibility
        # Compute p-p_ref for each vertex of the entire network
        pressure_difference_vertex = flownetwork.pressure - distensibility.pressure_ref
        # Compute p-p_ref for each edge that can change due to the distensibility. Take the mean pressure of adjacent
        # vertices to get the edge-based pressure difference. Only compute for distensible edges.
        pressure_difference_dist_edges = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[
            eids_dist]

        diameter_new = np.copy(flownetwork.diameter)
        # Compute the updated diameters
        diameter_new[eids_dist] = np.sqrt(np.square(distensibility.diameter_ref) *
                                          (1 + pressure_difference_dist_edges * distensibility.diameter_ref /
                                           (distensibility.e_modulus * distensibility.wall_thickness)))
        # Update diameters
        flownetwork.diameter = diameter_new
