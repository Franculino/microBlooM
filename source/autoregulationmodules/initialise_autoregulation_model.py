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

        # Direct stress defines as the transmural pressure
        # External pressure
        pressure_external = self._PARAMETERS["pressure_external"]
        # Assign a constant external pressure for the entire vasculature
        pressure_ext = np.ones(flownetwork.nr_of_vs) * pressure_external
        # Transmural pressure
        pressure_difference_vertex_baseline = autoregulation.pressure_baseline - pressure_ext
        autoregulation.direct_stress_baseline = .5 * np.sum(pressure_difference_vertex_baseline[flownetwork.edge_list], axis=1)[eids_auto]

        # Shear Stresses (wall shear stress, τ=(32*q*μ)/(π*d^3)), but we compute only τ=Δp*d becuase we are interested
        # in the ratio between current situation and baseline (see Eq.17 from Payne et al. 2023)
        # Pressure gradient
        pressure_vertex_baseline = autoregulation.pressure_baseline[flownetwork.edge_list]
        pressure_drop_edge_baseline = np.abs(pressure_vertex_baseline[:,0] - pressure_vertex_baseline[:,1])
        autoregulation.shear_stress_baseline = pressure_drop_edge_baseline[eids_auto] * autoregulation.diameter_baseline[eids_auto]

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


class AutoregulationModelInitialiseSherwinRelation(AutoregulationModelInitialise):

    def initialise_baseline_compliance(self, autoregulation, flownetwork):
        """
        Specify the compliance at the baseline according to p-A relation proposed by Sherwin et al. (2003)
        The expression for baseline compliance (C0) around the baseline radius (R0) can be derived from combining
        Laplace’s law for cylindrical vessels with the constitutive relation for linear elastic and incompressible
        wall material behaviour (see Eq.8 from Payne et al. 2023)
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation

        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Edge ids that are autoregulatory vessels (actively diameter changes are possible)
        eids_auto = autoregulation.eid_vessel_autoregulation

        radius_baseline = autoregulation.diameter_baseline[eids_auto]/2.

        if not np.size(radius_baseline) == np.size(autoregulation.e_modulus):
            sys.exit("Warning: Mismatch of array size for the autoregulation model")

        autoregulation.compliance_baseline = 3/2 * (np.pi * radius_baseline**3. * flownetwork.length[eids_auto]) / \
                                             (autoregulation.e_modulus * autoregulation.wall_thickness)



