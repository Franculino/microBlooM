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
    """
    Update diameters based on the autoregulation model proposed by Payne et al. 2023
    """

    def update_diameter(self, autoregulation, flownetwork):
        """
        Update diameters based on the autoregulation model proposed by Payne et al. 2023
        To model the interaction between the myogenic and the endothelial responses, we assume a linear feedback model
        for the vessel relative stiffness k at the steady state (see Eq.16).
        The vessel stiffness is then mapped into vessel compliance through a non-linear sigmoidal function.
        The arteriolar compliance is updated according to the feedback model, after which the vessel parameter β is
        updated according to β = 2 sqrt(L V0)/ C, where V0 is the vascular volume at the baseline.
        Then the p-A relation is updated too.
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_auto = autoregulation.eid_vessel_autoregulation
        # Specify the diameters at the reference state based on the tube law initialisation
        autoregulation.diameter_ref = flownetwork.diameter_ref[eids_auto]

        # Direct Stresses
        # Compute the relative direct stress for simulating myogenic responses
        # Direct stress defines as the transmural pressure --> The intraluminal pressure is approximated as
        # the average of the inlet and outlet pressures of each vessel
        # External pressure
        pressure_external = self._PARAMETERS["pressure_external"]
        # Assign a constant external pressure for the entire vasculature
        pressure_ext = np.ones(flownetwork.nr_of_vs) * pressure_external
        # Baseline direct stresses
        direct_stress_baseline = np.copy(autoregulation.direct_stress_baseline)
        # Current direct stresses
        pressure_difference_vertex = flownetwork.pressure - pressure_ext
        direct_stress = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[eids_auto]

        # Shear Stresses
        # Compute the relative shear stresses for simulating endothelial responses
        # Baseline direct stresses
        shear_stress_baseline = np.copy(autoregulation.shear_stress_baseline)
        # Current shear stresses
        pressure_vertex = flownetwork.pressure[flownetwork.edge_list]
        pressure_drop_edge = np.abs(pressure_vertex[:,0] - pressure_vertex[:,1])
        shear_stress = pressure_drop_edge[eids_auto] * (flownetwork.diameter*0.5)[eids_auto]

        autoregulation.direct_stress = direct_stress
        autoregulation.shear_stress = shear_stress

        relative_stiffness = 1. + autoregulation.sens_direct * ((direct_stress / direct_stress_baseline) - 1.) - \
                              autoregulation.sens_shear * ((shear_stress / shear_stress_baseline) - 1.)

        if not np.size(relative_stiffness[relative_stiffness<0.]) == 0:
            sys.exit("Negative relative stiffness")

        autoregulation.rel_stiffness = relative_stiffness

        inverse_stiffness = 1./relative_stiffness  # compliance is inversely related to stiffness

        # The vessel stiffness is then mapped into vessel compliance through a non-linear sigmoidal function.
        # The maximum and minimum changes in compliance are different to account for the fact that compliance is not
        # symmetrical about its baseline value
        # Constant parameters
        max_compliance = 10  # maximum changes in compliance
        min_compliance = 0.8  # minimum changes in compliance
        slope = 0.1  # central slope steepness
        # Relative compliance based on the non-linear sigmoidal function
        relative_compliance = np.ones(np.size(relative_stiffness))
        relative_compliance[relative_stiffness<1.] = 1. + max_compliance * np.tanh(1./slope * 1./max_compliance * (inverse_stiffness[relative_stiffness<1.] - 1.))
        relative_compliance[relative_stiffness>=1.] = 1. + min_compliance * np.tanh(1./slope * 1./min_compliance * (inverse_stiffness[relative_stiffness>=1.] - 1.))

        if not np.size(relative_compliance[relative_compliance < 0.]) == 0:
            sys.exit("Negative relative compliance")

        autoregulation.rel_compliance = relative_compliance

        # C = C_relative * C_baseline
        autoregulation.compliance = relative_compliance * autoregulation.compliance_baseline

        # second version
        diameter_predicted = np.copy(flownetwork.diameter)
        volume_base = np.pi * flownetwork.length[eids_auto] * np.square(autoregulation.diameter_baseline[eids_auto]*0.5)
        length = flownetwork.length[eids_auto]
        radius_ref = autoregulation.diameter_ref * 0.5
        constant = 2.*np.sqrt(np.pi*length*volume_base)/autoregulation.compliance
        radius = (direct_stress + constant*radius_ref) / constant
        diameter_predicted[eids_auto] = 2. * radius

        """
        # forth version
        diameter_predicted = np.copy(flownetwork.diameter)
        diam_ref = autoregulation.diameter_ref
        length = flownetwork.length[eids_auto]
        volume = np.pi * length * np.square(autoregulation.diameter_baseline[eids_auto] * 0.5)
        beta = 2.*np.sqrt(length*volume)/autoregulation.compliance
        area_ref = np.pi * np.square(diam_ref*0.5)
        diam = 2. * ((direct_stress + beta * np.sqrt(area_ref))/(beta * np.sqrt(np.pi)))
        diameter_predicted[eids_auto] = diam
        """
        """
        # third version
        diameter_predicted = np.copy(flownetwork.diameter)
        diam_ref = flownetwork.diameter_ref[eids_auto]
        length = flownetwork.length[eids_auto]
        diam = 0.5 * (diam_ref + np.sqrt(np.square(diam_ref) + (8. * direct_stress * autoregulation.compliance)/(np.pi * length)))
        diameter_predicted[eids_auto] = diam
        """
        """
        # first version
        diameter_predicted = np.copy(flownetwork.diameter)
        # Based on Sherwin et al. (2003) with β around the current state (poisson ratio=0.5)
        # d = 2 * ((2 * E * h * C) / (3* L * π))**(1/3)
        # autoregulation.wall_thickness = np.copy(flownetwork.diameter)[eids_auto] * 0.1
        diameter_predicted[eids_auto] = 2. * ((2. * autoregulation.e_modulus * autoregulation.wall_thickness *
                                               autoregulation.compliance) / (3. * np.pi * flownetwork.length[eids_auto]))**(1/3)

        # diameter_predicted[eids_auto] = 2. * np.sqrt((0.4 * autoregulation.e_modulus * autoregulation.compliance)/(3. * np.pi * flownetwork.length[eids_auto]))
        #
        # Based on Sherwin et al. (2003) with β around the baseline (poisson ratio=0.5)
        # d = 4/3 * (E * h * C) / (A_base * L)
        # area_baseline = np.pi * (np.copy(flownetwork.diameter)[eids_auto]*0.5)**2
        # autoregulation.wall_thickness = 0.1 * np.copy(flownetwork.diameter)[eids_auto]
        # diameter_predicted[eids_auto] = 4/3 * (autoregulation.e_modulus * autoregulation.wall_thickness * autoregulation.compliance) / (area_baseline[eids_auto] * flownetwork.length[eids_auto])

        # volume = np.pi * flownetwork.length[eids_auto] * (np.copy(autoregulation.diameter_baseline)[eids_auto]*0.5)**2
        # autoregulation.wall_thickness = 0.1 * np.copy(flownetwork.diameter)[eids_auto]
        # diameter_predicted[eids_auto] = 2. * ((2. * autoregulation.e_modulus * autoregulation.wall_thickness *
        #                                        autoregulation.compliance) / (3. * (np.pi * flownetwork.length[eids_auto]* volume)**(1/2)))**(1/2)
        """

        # relaxation function - d_i = (1-a)*d_i-1 + a*d_i
        diameter_new = np.copy(flownetwork.diameter)
        diameter_new[eids_auto] = (1. - autoregulation.alpha) * autoregulation.diameter_previous[eids_auto] + autoregulation.alpha * diameter_predicted[eids_auto]

        flownetwork.diameter = diameter_new


class AutoregulationModelDominikModel(AutoregulationModelUpdate):
    """
    Update diameters based on the autoregulation model according Dominik's approach
    """

    def update_diameter(self, autoregulation, flownetwork):
        """
        Update diameters based on the autoregulation model according Dominik's approach
        C = dV/dPt --> ΔV = C ΔPt
        In this way we avoid using the parameter β. --> I think this is not the correct way.
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_auto = autoregulation.eid_vessel_autoregulation
        # Specify the diameters at the reference state based on the tube law initialisation
        autoregulation.diameter_ref = flownetwork.diameter_ref[eids_auto]
        # Direct Stresses
        # Compute the relative direct stress for simulating myogenic responses
        # Direct stress defines as the transmural pressure --> The intraluminal pressure is approximated as
        # the average of the inlet and outlet pressures of each vessel
        # External pressure
        pressure_external = self._PARAMETERS["pressure_external"]
        # Assign a constant external pressure for the entire vasculature
        pressure_ext = np.ones(flownetwork.nr_of_vs) * pressure_external
        # Baseline direct stresses
        direct_stress_baseline = np.copy(autoregulation.direct_stress_baseline)
        # Current direct stresses
        pressure_difference_vertex = flownetwork.pressure - pressure_ext
        direct_stress = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[eids_auto]

        # Shear Stresses
        # Compute the relative shear stresses for simulating endothelial responses
        # Baseline direct stresses
        shear_stress_baseline = np.copy(autoregulation.shear_stress_baseline)
        # Current shear stresses
        pressure_vertex = flownetwork.pressure[flownetwork.edge_list]
        pressure_drop_edge = np.abs(pressure_vertex[:,0] - pressure_vertex[:,1])
        shear_stress = pressure_drop_edge[eids_auto] * (flownetwork.diameter*0.5)[eids_auto]

        autoregulation.direct_stress = direct_stress
        autoregulation.shear_stress = shear_stress

        relative_stiffness = 1. + autoregulation.sens_direct * ((direct_stress / direct_stress_baseline) - 1.) - \
                              autoregulation.sens_shear * ((shear_stress / shear_stress_baseline) - 1.)

        if not np.size(relative_stiffness[relative_stiffness<0.]) == 0:
            sys.exit("Negative relative stiffness")

        autoregulation.rel_stiffness = relative_stiffness

        inverse_stiffness = 1./relative_stiffness  # compliance is inversely related to stiffness

        # The vessel stiffness is then mapped into vessel compliance through a non-linear sigmoidal function.
        # The maximum and minimum changes in compliance are different to account for the fact that compliance is not
        # symmetrical about its baseline value
        # Constant parameters
        max_compliance = 10  # maximum changes in compliance
        min_compliance = 0.8  # minimum changes in compliance
        slope = 0.04  # central slope steepness
        # Relative compliance based on the non-linear sigmoidal function
        relative_compliance = np.ones(np.size(relative_stiffness))
        relative_compliance[relative_stiffness<1.] = 1. + max_compliance * np.tanh(1./slope * 1./max_compliance * (inverse_stiffness[relative_stiffness<1.] - 1.))
        relative_compliance[relative_stiffness>=1.] = 1. + min_compliance * np.tanh(1./slope * 1./min_compliance * (inverse_stiffness[relative_stiffness>=1.] - 1.))

        if not np.size(relative_compliance[relative_compliance < 0.]) == 0:
            sys.exit("Negative relative compliance")

        autoregulation.rel_compliance = relative_compliance

        # C = C_relative * C_baseline
        autoregulation.compliance = relative_compliance * autoregulation.compliance_baseline

        # our version
        diameter_predicted = np.copy(flownetwork.diameter)

        if autoregulation.iteration == 0:
            delta_transmural_pressure = autoregulation.direct_stress_baseline - direct_stress
            delta_volume = autoregulation.compliance * delta_transmural_pressure
            previous_volume = np.pi * flownetwork.length[eids_auto] * np.square(autoregulation.diameter_baseline[eids_auto]*0.5)
            current_volume = previous_volume + delta_volume
        else:
            delta_transmural_pressure = autoregulation.direct_stress_previous - direct_stress
            delta_volume = autoregulation.compliance * delta_transmural_pressure
            previous_volume = autoregulation.volume_previous
            current_volume = previous_volume + delta_volume

        diam = 2. * np.sqrt(current_volume/(np.pi * flownetwork.length[eids_auto]))
        diameter_predicted[eids_auto] = diam

        # relaxation function - d_i = (1-a)*d_i-1 + a*d_i
        # diameter_new = np.copy(flownetwork.diameter)
        # diameter_new[eids_auto] = (1. - autoregulation.alpha) * autoregulation.diameter_previous[eids_auto] + autoregulation.alpha * diameter_predicted[eids_auto]
        # flownetwork.diameter = diameter_new

        flownetwork.diameter = diameter_predicted

        autoregulation.volume_previous = current_volume
        autoregulation.direct_stress_previous = direct_stress


class AutoregulationModelOurApproach(AutoregulationModelUpdate):
    """
    Update diameters by adjusting the autoregulation model proposed by Payne et al. 2023
    """

    def update_diameter(self, autoregulation, flownetwork):
        """
        Update diameters by adjusting the autoregulation model proposed by Payne et al. 2023
        To model the interaction between the myogenic and the endothelial responses, we assume a linear feedback model
        for the vessel relative stiffness k at the steady state (see Eq.16).
        The vessel stiffness is then mapped into vessel compliance through a non-linear sigmoidal function.
        The arteriolar compliance is updated according to the feedback model, after which the vessel parameter β is
        updated according to β = 2 sqrt(L V)/ C, where V is the vascular volume at the current state.
        Then the p-A relation is updated too.
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_auto = autoregulation.eid_vessel_autoregulation
        # Specify the diameters at the reference state based on the tube law initialisation
        autoregulation.diameter_ref = flownetwork.diameter_ref[eids_auto]

        # Direct Stresses
        # Compute the relative direct stress for simulating myogenic responses
        # Direct stress defines as the transmural pressure --> The intraluminal pressure is approximated as
        # the average of the inlet and outlet pressures of each vessel
        # External pressure
        pressure_external = self._PARAMETERS["pressure_external"]
        # Assign a constant external pressure for the entire vasculature
        pressure_ext = np.ones(flownetwork.nr_of_vs) * pressure_external
        # Baseline direct stresses
        direct_stress_baseline = np.copy(autoregulation.direct_stress_baseline)
        # Current direct stresses
        pressure_difference_vertex = flownetwork.pressure - pressure_ext
        direct_stress = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[eids_auto]
        autoregulation.direct_stress = direct_stress

        # Shear Stresses
        # Compute the relative shear stresses for simulating endothelial responses
        # Baseline direct stresses
        shear_stress_baseline = np.copy(autoregulation.shear_stress_baseline)
        # Current shear stresses
        pressure_vertex = flownetwork.pressure[flownetwork.edge_list]
        pressure_drop_edge = np.abs(pressure_vertex[:,0] - pressure_vertex[:,1])
        shear_stress = pressure_drop_edge[eids_auto] * (flownetwork.diameter)[eids_auto]
        # Set the minimum non-zero value to zero shear stresses
        if np.size(shear_stress_baseline[shear_stress_baseline == 0.])/autoregulation.nr_of_edge_autoregulation*100 > 0.5:
            sys.exit("Warring Error: Suspicious many zero shear stresses detected at the current state.")
        shear_stress[shear_stress == 0.] = np.min(shear_stress[shear_stress != 0.])
        autoregulation.shear_stress = shear_stress

        if autoregulation.iteration <= 20:
            # Wall shear stress, τ=(32*q*μ)/(π*d^3) in Pa
            shear_stress_Pa = ((32. * np.abs(flownetwork.flow_rate) * self._PARAMETERS["mu_plasma"] * flownetwork.mu_rel) /
                               (np.pi * np.power(flownetwork.diameter, 3.)))[eids_auto]
            # According literature, there is a shear stress threshold for producing NO
            sens_shear = np.ones(autoregulation.nr_of_edge_autoregulation, dtype=float) * self._PARAMETERS["sensitivity_shear_stress"]
            sens_shear[shear_stress_Pa <= 0.1] = 0.
            autoregulation.sens_shear_previous = sens_shear

        autoregulation.sens_shear = autoregulation.sens_shear_previous

        autoregulation.sens_direct = np.ones(autoregulation.nr_of_edge_autoregulation, dtype=float) * self._PARAMETERS["sensitivity_direct_stress"]

        relative_stiffness = 1. + autoregulation.sens_direct * ((direct_stress / direct_stress_baseline) - 1.) - \
                             autoregulation.sens_shear * ((shear_stress / shear_stress_baseline) - 1.)

        if not np.size(relative_stiffness[relative_stiffness <= 0.]) == 0:
            print("Negative relative stiffness - Number of vessel:", np.size(relative_stiffness[relative_stiffness <= 0.]))
            relative_stiffness[relative_stiffness <= 0.] = 0.001

        autoregulation.rel_stiffness = relative_stiffness

        inverse_stiffness = 1./relative_stiffness  # compliance is inversely related to stiffness

        # The vessel stiffness is then mapped into vessel compliance through a non-linear sigmoidal function.
        # The maximum and minimum changes in compliance are different to account for the fact that compliance is not
        # symmetrical about its baseline value
        # Constant parameters
        max_compliance = 10  # maximum changes in compliance
        min_compliance = 0.9  # minimum changes in compliance
        slope = 0.1  # central slope steepness
        # Relative compliance based on the non-linear sigmoidal function
        relative_compliance = np.ones(np.size(relative_stiffness))
        relative_compliance[relative_stiffness<1.] = 1. + max_compliance * np.tanh(1./slope * 1./max_compliance * (inverse_stiffness[relative_stiffness<1.] - 1.))
        relative_compliance[relative_stiffness>=1.] = 1. + min_compliance * np.tanh(1./slope * 1./min_compliance * (inverse_stiffness[relative_stiffness>=1.] - 1.))

        if not np.size(relative_compliance[relative_compliance < 0.]) == 0:
            sys.exit("Negative relative compliance")

        autoregulation.rel_compliance = relative_compliance

        # C = C_relative * C_baseline
        autoregulation.compliance = relative_compliance * autoregulation.compliance_baseline

        diameter_predicted = np.copy(flownetwork.diameter)
        diam_ref = autoregulation.diameter_ref
        length = flownetwork.length[eids_auto]
        diam = 0.5 * (diam_ref + np.sqrt(np.square(diam_ref) + (8. * direct_stress * autoregulation.compliance)/(np.pi * length)))
        diameter_predicted[eids_auto] = diam

        # relaxation function - d_i = (1-a)*d_i-1 + a*d_i
        diameter_new = np.copy(flownetwork.diameter)
        diameter_new[eids_auto] = (1. - autoregulation.alpha) * autoregulation.diameter_previous[eids_auto] + autoregulation.alpha * diameter_predicted[eids_auto]

        flownetwork.diameter = diameter_new

        if True in (diameter_new < .5 * autoregulation.diameter_baseline) or True in (diameter_new > 2.5 * autoregulation.diameter_baseline):
            sys.exit("Error: Suspicious current diameters (compared to baseline diameter) detected.")
