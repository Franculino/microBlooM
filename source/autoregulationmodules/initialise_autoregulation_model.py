from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class AutoregulationModelInitialise(ABC):
    """
    Abstract base class for updating the diameters
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of AutoregulationModelInitialise.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    def initialise_baseline_stresses(self, autoregulation, flownetwork):
        """
        Specify the direct (σ) and shear (τ) stress at the baseline
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Edge ids that are autoregulatory vessels (actively diameter changes are possible)
        eids_auto = autoregulation.eid_vessel_autoregulation

        # Shear Stresses (wall shear stress, τ=(32*q*μ)/(π*d^3)), but we compute only τ=Δp*d becuase we are interested
        # in the ratio between current situation and baseline (see Eq.17 from Payne et al., 2023)
        # Pressure gradient along vessels
        pressure_vertex_baseline = autoregulation.pressure_baseline[flownetwork.edge_list]
        pressure_drop_edge_baseline = np.abs(pressure_vertex_baseline[:,0] - pressure_vertex_baseline[:,1])
        shear_stress_baseline = pressure_drop_edge_baseline[eids_auto] * (autoregulation.diameter_baseline)[eids_auto]

        # Set the minimum non-zero value to zero shear stresses
        if np.size(shear_stress_baseline[shear_stress_baseline == 0.])/autoregulation.nr_of_edge_autoregulation*100 > 0.5:
            sys.exit("Warring Error: Suspicious many zero shear stresses detected at the baseline.")
        shear_stress_baseline[shear_stress_baseline == 0.] = np.min(shear_stress_baseline[shear_stress_baseline != 0.])
        autoregulation.shear_stress_baseline = shear_stress_baseline

        # Direct stress defines as the transmural pressure
        # External pressure
        pressure_external = self._PARAMETERS["pressure_external"]
        # Assign a constant external pressure for the entire vasculature
        pressure_ext = np.ones(flownetwork.nr_of_vs) * pressure_external
        # Transmural pressure
        pressure_difference_vertex_baseline = autoregulation.pressure_baseline - pressure_ext
        autoregulation.direct_stress_baseline = .5 * np.sum(pressure_difference_vertex_baseline[flownetwork.edge_list], axis=1)[eids_auto]


    @abstractmethod
    def initialise_baseline_compliance(self, autoregulation, flownetwork):
        """
        Specify the compliance at the baseline according to p-A relation
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class AutoregulationModelInitialiseNothing(AutoregulationModelInitialise):

    def initialise_baseline_compliance(self, autoregulation, flownetwork):
        """
        Do not specify the compliance at the baseline
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class AutoregulationModelInitialisePayneRelation(AutoregulationModelInitialise):
    """
    Compute the compliance at the baseline according to the relation proposed by Payne et al. (2023)
    """

    def initialise_baseline_compliance(self, autoregulation, flownetwork):
        """
        Compute the compliance at the baseline according to p-A relation proposed by Sherwin et al. (2003)
        The expression for baseline compliance (C0) around the baseline radius (R0) can be derived from combining
        Laplace’s law for cylindrical vessels with the constitutive relation for linear elastic and incompressible
        wall material behaviour (see Eq.8 from Payne et al. 2023) --> C0 = 3/2 (π L R0^3) / (E h)
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Compute the vascular compliance at the baseline
        # Edge ids that are autoregulatory vessels (actively diameter changes are possible)
        eids_auto = autoregulation.eid_vessel_autoregulation

        radius_baseline = autoregulation.diameter_baseline[eids_auto]/2.
        length = flownetwork.length[eids_auto]
        e_modulus = flownetwork.e_modulus[eids_auto]
        wall_thickness = flownetwork.wall_thickness[eids_auto]

        compliance_baseline = 3 / 2 * (np.pi * (radius_baseline**3.) * length) / (e_modulus * wall_thickness)
        autoregulation.compliance_baseline = np.copy(compliance_baseline)


class AutoregulationModelInitialiseOurRelation(AutoregulationModelInitialise):
    """
    Compute the compliance at the baseline using the definition C = dV/dPt based on the p-A relation proposed by
    Sherwin et al. (2023)
    """

    def initialise_baseline_compliance(self, autoregulation, flownetwork):
        """
        Compute the compliance at the baseline using the definition C = dV/dP based on the p-A relation proposed by
        Sherwin et al. (2023). By definition, the vessel compliance (C) in the passive (non-autoregulating) case is
        given by C = dV/dPt. The expression for baseline compliance (C0) around the baseline radius (R0) can be derived
        from combining Laplace’s law for cylindrical vessels with the constitutive relation for linear elastic and
        incompressible wall material behaviour. --> C0 = 3/2 (π L R0 Rref^2) / (E h)
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Compute the vascular compliance at the baseline
        # Edge ids that are autoregulatory vessels (actively diameter changes are possible)
        eids_auto = autoregulation.eid_vessel_autoregulation

        radius_baseline = autoregulation.diameter_baseline[eids_auto]/2.
        radius_ref = flownetwork.diameter_ref[eids_auto]/2.
        length = flownetwork.length[eids_auto]
        e_modulus = flownetwork.e_modulus[eids_auto]
        wall_thickness = flownetwork.wall_thickness[eids_auto]

        compliance_baseline = 3 / 2 * (np.pi * length * radius_baseline * np.square(radius_ref)) / \
                              (e_modulus * wall_thickness)
        autoregulation.compliance_baseline = np.copy(compliance_baseline)


