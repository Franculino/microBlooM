from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class DistensibilityLawInitialise(ABC):
    """
    Abstract base class for initialiasing the distensibility law
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of DistensibilityLawInitialisation.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
        """
        Specify the reference pressures and diameters for each vessel
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class DistensibilityLawUpdate(ABC):
    """
    Abstract base class for updating the diameters
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of DistensibilityLawInitialisation.
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


class DistensibilityInitialiseNothing(DistensibilityLawInitialise):
    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
        """
        Do not update any diameters based on vessel distensibility
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class DistensibilityLawPassiveReferenceBaselinePressure(DistensibilityLawInitialise):

    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
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


class DistensibilityLawPassiveReferenceConstantExternalPressureSherwin(DistensibilityLawInitialise):

    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
        """
        Set the reference pressure to the constant external pressure. Compute a reference diameter for each
        vessel based on the specified reference pressure and the baseline diameters & pressures.
        Based on a non-linear p-A ralation proposed by Sherwin et al. (2003)
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


class DistensibilityLawPassiveReferenceConstantExternalPressureUrquiza(DistensibilityLawInitialise):

    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
        """
        Set the reference pressure to the constant external pressure. Compute a reference diameter for each
        vessel based on the specified reference pressure and the baseline diameters & pressures.
        Based on a non-linear p-A relation proposed by Urquiza et al. (2006)
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
        kappa = 2 * distensibility.e_modulus * distensibility.wall_thickness / pressure_difference_edge
        diameter_ref = .5 * (-kappa + np.sqrt(np.square(kappa) + 4 * kappa * flownetwork.diameter[eids_dist]))

        if True in (diameter_ref < .5 * flownetwork.diameter[eids_dist]):
            sys.exit("Error: Suspicious small reference diameter (compared to baseline diameter) detected. ")

        distensibility.diameter_ref = diameter_ref




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
                                          (1 - pressure_difference_dist_edges * distensibility.diameter_ref /
                                           (distensibility.e_modulus * distensibility.wall_thickness)))
        # Update diameters
        flownetwork.diameter = diameter_new
