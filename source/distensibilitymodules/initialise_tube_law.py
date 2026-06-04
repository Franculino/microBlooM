from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class TubeLawInitialision(ABC):
    """
    Abstract base class for initialiasing the tube law which is a relationship that relates the changes in the
    transmural pressure(Pt = P − Pext) to those in the cross-sectional (CS) area. The pressure-area relation is for
    the flow in elastic tubes. This approach is based on the linear theory of elasticity.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of TubeLawInitialision.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def initialise_ref_state(self, flownetwork):
        """
        Specify the reference pressures and diameters for each vessel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class TubeLawInitialisionNothing(TubeLawInitialision):
    def initialise_ref_state(self, flownetwork):
        """
        Do not update any diameters based on vessel distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class TubeLawPassiveReferenceBaselinePressure(TubeLawInitialision):
    """
    Define the reference state based on the current baseline values
    """
    def initialise_ref_state(self, flownetwork):
        """
        Specify the reference pressure and diameter to the current baseline values (at time of initialisation)
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.pressure_ref = np.copy(flownetwork.pressure)  # External pressure corresponds to baseline pres
        # Reference diameter corresponds to baseline diameter
        flownetwork.diameter_ref = np.copy(flownetwork.diameter)


class TubeLawPassiveReferenceConstantExternalPressureSherwin(TubeLawInitialision):
    """
    Define the reference state based on a non-linear p-A ralation proposed by Sherwin et al. (2003).
    """
    def initialise_ref_state(self, flownetwork):
        """
        Set the reference pressure to the constant external pressure. Compute a reference diameter for each
        vessel based on the specified reference pressure and the baseline diameters & pressures.
        Based on a non-linear p-A ralation proposed by Sherwin et al. (2003).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # External pressure
        flownetwork.pressure_external = self._PARAMETERS["pressure_external"]
        # Assign a constant external pressure for the entire vasculature
        flownetwork.pressure_ref = np.ones(flownetwork.nr_of_vs)*flownetwork.pressure_external

        # Compute the reference diameter based on the constant external pressure and the baseline diameters & pressures.
        # Reference diameter does not correspond to the baseline diameter.
        pressure_difference_vertex = flownetwork.pressure - flownetwork.pressure_ref
        pressure_difference_edge = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)

        # Solve quadratic formula for diameter ref {-b + sqrt(b^2-4*a*c)} (other solution is invalid)
        kappa = 2 * flownetwork.e_modulus * flownetwork.wall_thickness / (
                    (pressure_difference_edge) * (1. - np.square(flownetwork.nu)))
        diameter_ref = .5 * (-kappa + np.sqrt(np.square(kappa) + 4 * kappa * flownetwork.diameter))

        if True in (diameter_ref < .5 * flownetwork.diameter):
            sys.exit("Error: Suspicious small reference diameter (compared to baseline diameter) detected. ")

        flownetwork.diameter_ref = diameter_ref


class TubeLawPassiveReferenceConstantExternalPressurePayne(TubeLawInitialision):
    """
    Define the reference state based on the relation (without any assumptions) suggested by Payne et al. 2023
    """
    def initialise_ref_state(self, flownetwork):
        """
        Set the reference pressure to the constant external pressure. Compute a reference diameter for each
        vessel based on the specified reference pressure and the baseline diameters & pressures.
        Based on the relation (without any assumptions) suggested by Payne et al. 2023 (see Eq. A.4 in Appendix A)
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # External pressure
        flownetwork.pressure_external = self._PARAMETERS["pressure_external"]
        # Assign a constant external pressure for the entire vasculature
        flownetwork.pressure_ref = np.ones(flownetwork.nr_of_vs)*flownetwork.pressure_external

        # Compute the reference diameter based on the constant external pressure and the baseline diameters & pressures.
        # Reference diameter does not correspond to the baseline diameter.
        pressure_difference_vertex = flownetwork.pressure - flownetwork.pressure_ref
        pressure_difference_edge = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)

        # Based on the relation suggested by Payne et al. 2023 --> Eq. A.4 in Appendix A
        radius_baseline = flownetwork.diameter * 0.5
        radius_ref = (flownetwork.e_modulus * flownetwork.wall_thickness * radius_baseline) / \
                ((1. - np.square(flownetwork.nu)) * pressure_difference_edge * radius_baseline +
                 (flownetwork.e_modulus * flownetwork.wall_thickness))

        diameter_ref = radius_ref * 2.

        if True in (diameter_ref < .5 * flownetwork.diameter):
            sys.exit("Error: Suspicious small reference diameter (compared to baseline diameter) detected. ")

        flownetwork.diameter_ref = diameter_ref


