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
        Constructor of Distensibility Law.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def initialise_distensibility_law(self, distensibility, flownetwork):
        """
        Specify the reference pressures and diameters for each vessel
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def update_diameter(self, distensibility, flownetwork):
        """
        Update the diameters based on a distensibility law
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class DistensibilityNothing(DistensibilityLaw):
    def initialise_distensibility_law(self, distensibility, flownetwork):
        """
        Do not update any diameters based on vessel distensibility
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    def update_diameter(self, distensibility, flownetwork):
        """
        Do not update any diameters based on vessel distensibility
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class DistensibilityLawPassiveLinear(DistensibilityLaw):

    @abstractmethod
    def initialise_distensibility_law(self, distensibility, flownetwork):
        """
        Specify the reference pressure and diameter
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    def update_diameter(self, distensibility, flownetwork):
        """
        Update the diameters based on a linear, passive distensibility law
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

        flownetwork.diameter = diameter_new


class DistensibilityLawPassiveLinearReferenceBaselinePressure(DistensibilityLawPassiveLinear):

    def initialise_distensibility_law(self, distensibility, flownetwork):
        """
        Specify the reference pressure and diameter to the current baseline values (at time of initialisation)
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        distensibility.pressure_ref = np.copy(flownetwork.pressure)  # External pressure corresponds to baseline pres
        # Reference diameter corresponds to baseline diameter
        distensibility.diameter_ref = np.copy(flownetwork.diameter)[distensibility.eid_vessel_distensibility]


class DistensibilityLawPassiveLinearReferenceConstantExternalPressure(DistensibilityLawPassiveLinear):

    def initialise_distensibility_law(self, distensibility, flownetwork):
        """
        Set the reference pressure to the constant external pressure. Compute a reference diameter for each
        vessel based on the specified reference pressure and the baseline diameters & pressures.
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # External pressure
        distensibility.pressure_external = self._PARAMETERS["pressure_external"]
        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_dist = distensibility.eid_vessel_distensibility
        # Assign a constant external pressure for the entire vasculature
        distensibility.pressure_ref = np.ones(flownetwork.nr_of_vs)*distensibility.pressure_external

        # Compute the reference diameter based on the constant external pressure and the baseline diameters & pressures.
        # Reference diameter does not correspond to the baseline diameter.
        pressure_difference_vertex = flownetwork.pressure - distensibility.pressure_ref
        pressure_difference_edge = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[
            eids_dist]

        # Solve quadratic formula for diameter ref {-b + sqrt(b^2-4*a*c)} (other solution is invalid)
        kappa = 2 * distensibility.e_modulus * distensibility.wall_thickness / (
                    (pressure_difference_edge) * (1. - np.square(distensibility.nu)))
        diameter_ref = .5 * (-kappa + np.sqrt(np.square(kappa) + 4 * kappa * flownetwork.diameter[eids_dist]))

        if True in (diameter_ref < .5 * flownetwork.diameter[eids_dist]):
            sys.exit("Error: Suspicious small reference diameter (compared to baseline diameter) detected. ")

        distensibility.diameter_ref = diameter_ref
