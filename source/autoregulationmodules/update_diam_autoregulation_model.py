from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class AutoregulationModelUpdate(ABC):
    """
    Abstract base class for updating the diameters for the autorequlatory vessels
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of AutoregulationModelUpdate.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def update_diameter(self, autoregulation, flownetwork):
        """
        Update the diameters based on the autoregulation model
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class AutoregulationModelUpdateNothing(AutoregulationModelUpdate):

    def update_diameter(self, autoregulation, flownetwork):
        """
        Do not update any diameters based on the autoregulation model
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class AutoregulationModelUpdatePayne2023(AutoregulationModelUpdate):

    def update_diameter(self, autoregulation, flownetwork):
        """
        Update diameters based on the autoregulation model proposed by Payne et al. 2023
        To model the interaction between the myogenic and the endothelial responses, we assume a linear feedback model
        for the vessel relative stiffness k at the steady state (see Eq.16).
        The vessel stiffness is then mapped into vessel compliance through a non-linear sigmoidal function.
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_auto = autoregulation.eid_vessel_autoregulation

        # Direct Stresses
        # Compute the relative direct stress for simulating myogenic responses
        # Direct stress defines as the transmural pressure
        # External pressure
        pressure_external = self._PARAMETERS["pressure_external"]
        # Assign a constant external pressure for the entire vasculature
        pressure_ext = np.ones(flownetwork.nr_of_vs) * pressure_external
        # Baseline direct stresses
        direct_stress_baseline = autoregulation.direct_stress_baseline
        # Current direct stresses
        pressure_difference_vertex = flownetwork.pressure - pressure_ext
        direct_stress = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[eids_auto]

        # Shear Stresses
        # Compute the relative shear stresses for simulating endothelial responses
        # Baseline direct stresses
        shear_stress_baseline = autoregulation.shear_stress_baseline
        # Current shear stresses
        pressure_vertex = flownetwork.pressure[flownetwork.edge_list]
        pressure_drop_edge = np.abs(pressure_vertex[:,0] - pressure_vertex[:,1])
        shear_stress = pressure_drop_edge[eids_auto] * flownetwork.diameter[eids_auto]

        relative_stiffness = 1. + autoregulation.sens_direct * (direct_stress / direct_stress_baseline - 1.) - \
                              autoregulation.sens_shear * (shear_stress / shear_stress_baseline - 1.)

        if not np.size(relative_stiffness[relative_stiffness<0.]) == 0:
            sys.exit("Negative relative stiffness")


        # print("relative_stiffness")
        # print(relative_stiffness)
        # print("direct_stress")
        # print(autoregulation.sens_direct * (direct_stress / direct_stress_baseline - 1.))
        # print("shear_stress")
        # print(autoregulation.sens_shear * (shear_stress / shear_stress_baseline - 1.))
        # print("rel. change pressure")
        # print(flownetwork.pressure / autoregulation.pressure_baseline - 1.)

        inverse_stiffness = 1./relative_stiffness  # compliance s inversely related to stiffness

        # The vessel stiffness is then mapped into vessel compliance through a non-linear sigmoidal function.
        # The maximum and minimum changes in compliance are different to account for the fact that compliance is not
        # symmetrical about its baseline value
        # Constant parameters
        max_compliance = 10  # maximum changes in compliance
        min_compliance = 0.8  # minimum changes in compliance
        slope = 0.1  # central slope steepness
        # Relative compliance based on the non-linear sigmoidal function
        relative_compliance = np.ones(np.size(relative_stiffness))
        relative_compliance[inverse_stiffness>=1.] = 1. + max_compliance * np.tanh(1./slope * 1./max_compliance * (inverse_stiffness[inverse_stiffness>=1.] - 1.))
        relative_compliance[inverse_stiffness<1.] = 1. + min_compliance * np.tanh(1./slope * 1./min_compliance * (inverse_stiffness[inverse_stiffness<1.] - 1.))


        # C = C_relative * C_baseline
        autoregulation.compliance = relative_compliance * autoregulation.compliance_baseline

        print("relative_compliance")
        print(relative_compliance)

        auto_diameter_new = np.copy(flownetwork.diameter)
        # Based on Sherwin et al. (2003) with β around the current state (poisson ratio=0.5)
        # d = 2 * ((2 * E * h * C) / (3* L * π))**(1/3)
        auto_diameter_new[eids_auto] = 2. * ((2. * autoregulation.e_modulus * autoregulation.wall_thickness *
                                              autoregulation.compliance) / (3. * np.pi * flownetwork.length[eids_auto]))**(1/3)
        #
        # Based on Sherwin et al. (2003) with β around the baseline (poisson ratio=0.5)
        # d = 4/3 * (E * h * C) / (A_base * L)
        # area_baseline = np.pi * (autoregulation.diameter_baseline/2)**2
        # auto_diameter_new[eids_auto] = 4/3 * (autoregulation.e_modulus * autoregulation.wall_thickness * autoregulation.compliance) / (area_baseline[eids_auto] * flownetwork.length[eids_auto])
        flownetwork.diameter = auto_diameter_new


class AutoregulationModelUpdateUrsino1997(AutoregulationModelUpdate):

    def update_diameter(self, autoregulation, flownetwork):
        """
        Update diameters based on the autoregulation model proposed by Payne et al. 2023
        To model the interaction between the myogenic and the endothelial responses, we assume a linear feedback model
        for the vessel relative stiffness k at the steady state (see Eq.16).
        The vessel stiffness is then mapped into vessel compliance through a non-linear sigmoidal function.
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_auto = autoregulation.eid_vessel_autoregulation

        x = flownetwork.flow_rate[eids_auto] / autoregulation.flow_rate_baseline[eids_auto] - 1

        dCa1 = 0.75
        dCa2 = 0.075
        Can = 0.15

        G = 1.5
        compliance = np.ones(np.size(eids_auto))
        compliance[x < 0] = ((Can + dCa1 / 2) + (Can - dCa1 / 2) * np.exp(G * x[x < 0] / (dCa1 / 4))) / (1 + np.exp(G * x[x < 0] / (dCa1 / 4)))
        compliance[x >= 0] = ((Can + dCa2 / 2) + (Can - dCa2 / 2) * np.exp(G * x[x >= 0] / (dCa2 / 4))) / (1 + np.exp(G * x[x >= 0] / (dCa2 / 4)))

        # C = C_relative * C_baseline
        autoregulation.compliance = compliance * 7.50061683e-9


        auto_diameter_new = np.copy(flownetwork.diameter)

        auto_diameter_new[eids_auto] = 2. * ((2. * autoregulation.e_modulus * autoregulation.wall_thickness *
                                              autoregulation.compliance) / (3. * np.pi * flownetwork.length[eids_auto]))**(1/3)

        print(autoregulation.compliance)
        print(auto_diameter_new[eids_auto])
        flownetwork.diameter = auto_diameter_new