from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys
# todo: this is a construction site


class ParameterSpace(ABC):
    """
    Abstract base class for the implementations specific to the chosen parameter space
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of ParameterSpace.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def initialise_parameters(self, inversemodel, flownetwork):
        """
        Initialises the parameter space
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def update_state(self, inversemodel, flownetwork):
        """
        Updates the system state of the flow network (such as diameters, transmissibilities, boundary pressures)
        based on the current parameter value
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def get_df_dalpha(self, inversemodel, flownetwork):
        """
        Computes the partial derivative of the cost function with respect to the parameter
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Partial derivative of f with respect to all parameters
        :rtype: 1d numpy array
        """

    @abstractmethod
    def _get_d_transmiss_d_alpha(self, inversemodel, flownetwork):
        pass



class ParameterSpaceRelativeDiameter(ParameterSpace):
    """
    Class for a parameter space that includes all vessel diameters relative to baseline
    """

    def initialise_parameters(self, inversemodel, flownetwork):
        """
        Initialises the parameter space, if relative diameters are tuned
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        inversemodel.diameter_baselinevalue = flownetwork.diameter[inversemodel.edge_param_eid]  # same length as alpha
        inversemodel.alpha = np.ones(inversemodel.nr_of_edge_parameters)
        inversemodel.alpha_prime = np.ones(inversemodel.nr_of_edge_parameters)

    def update_state(self, inversemodel, flownetwork):
        """
        Updates the system state of the flow network (here: diameter) based on the
        current parameter value (here: alpha = d/d_base)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.diameter[inversemodel.edge_param_eid] = inversemodel.diameter_baselinevalue * inversemodel.alpha

    def get_df_dalpha(self, inversemodel, flownetwork):
        """
        Computes the partial derivative of the cost function with respect to the parameter (relative diameter)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Partial derivative of f with respect to all parameters
        :rtype: 1d numpy array
        """

        # Initialise derivative d f / d alpha
        df_dalpha = np.zeros(inversemodel.nr_of_edge_parameters)

        # Compute all partial derivatives with respect to entire parameter vector
        d_flowrate_d_alpha = self._get_d_flowrate_d_alpha(inversemodel, flownetwork)  # d q_ij / d alpha_ij
        d_velocity_d_alpha = self._get_d_velocity_d_alpha(inversemodel, flownetwork)  # d u_ij / d alpha_ij

        # Type of target value (1: Flow rate, 2: Velocity,
        # True if current target value is of the respective type
        is_target_type_1 = np.in1d(inversemodel.edge_constraint_type, 1)  # constraint type 1: Flow rate constraint
        is_target_type_2 = np.in1d(inversemodel.edge_constraint_type, 2)  # constraint type 2: Velocity constraint

        # Identify current simulated value, and corresponding min and max target values
        val_current = flownetwork.flow_rate[inversemodel.edge_constraint_eid]
        val_min_tar = (inversemodel.edge_constraint_value - inversemodel.edge_constraint_range_pm)
        val_max_tar = (inversemodel.edge_constraint_value + inversemodel.edge_constraint_range_pm)

        # Difference between simulated and target (min, max) value (of all target edges)
        val_difference = np.zeros(inversemodel.nr_of_edge_constraints)
        val_difference[val_current > val_max_tar] = (val_current - val_max_tar)[val_current > val_max_tar]
        val_difference[val_current < val_min_tar] = (val_current - val_min_tar)[val_current < val_min_tar]

        # Target type 1 (flow rate)
        # Find the parameter and target id which corresponds to the edge with the target value
        mask_target_space_type_1 = np.logical_and(np.in1d(inversemodel.edge_constraint_eid, inversemodel.edge_param_eid), is_target_type_1)
        mask_param_space_type_1_precise = np.in1d(inversemodel.edge_param_eid, inversemodel.edge_constraint_eid[is_target_type_1])

        q_difference = val_difference[mask_target_space_type_1]  # q_simulated - q_target/q_min/q_max
        sigma_flow = inversemodel.edge_constraint_sigma[mask_target_space_type_1]

        # Partial derivative with respect to parameters; all flow rate terms
        df_dalpha[mask_param_space_type_1_precise] = 2. * q_difference / np.square(sigma_flow) * d_flowrate_d_alpha[mask_target_space_type_1]

        # Target type 2 (velocity)
        # Find the parameter and target id which corresponds to the edge with the target value
        mask_target_space_type_2 = np.logical_and(np.in1d(inversemodel.edge_constraint_eid, inversemodel.edge_param_eid), is_target_type_2)
        mask_param_space_type_2_precise = np.in1d(inversemodel.edge_param_eid, inversemodel.edge_constraint_eid[is_target_type_2])

        u_difference = val_difference[mask_target_space_type_2]  # u_simulated - u_target/u_min/u_max
        sigma_vel = inversemodel.edge_constraint_sigma[mask_target_space_type_2]

        # Partial derivative with respect to parameters; all velocity terms
        df_dalpha[mask_param_space_type_2_precise] = 2. * u_difference / np.square(sigma_vel) * d_velocity_d_alpha[
            mask_target_space_type_2]


    def _get_d_flowrate_d_alpha(self, inversemodel, flownetwork):

        edge_list_params = flownetwork.edge_list[inversemodel.edge_param_eid, :]

        # Return partial derivative d flowrate_ij / d alpha_ij (for all edges with parameters)
        return (flownetwork.pressure[edge_list_params[:, 0]] - flownetwork.pressure[
            edge_list_params[:, 1]]) * self._get_d_transmiss_d_alpha(inversemodel, flownetwork)

    def _get_d_velocity_d_alpha(self, inversemodel, flownetwork):

        hd_param_es = flownetwork.hd[inversemodel.edge_param_eid]
        ht_param_es = flownetwork.ht[inversemodel.edge_param_eid]
        diam_param_es = flownetwork.diameter[inversemodel.edge_param_eid]
        flowrate_param_es = flownetwork.flow_rate[inversemodel.edge_param_eid]

        d_flowrate_d_alpha = self._get_d_flowrate_d_alpha(inversemodel, flownetwork)

        return 4. / np.pi * hd_param_es / ht_param_es * (
                    d_flowrate_d_alpha / np.square(diam_param_es) - 2. * flowrate_param_es / (
                        np.square(inversemodel.diameter_baselinevalue) * np.power(inversemodel.alpha, 3)))

    def _get_d_transmiss_d_alpha(self, inversemodel, flownetwork):

        mu_plasma = self._PARAMETERS["mu_plasma"]
        # Lengths and mu_rel of edges that are parameters
        length_param_es = flownetwork.length[inversemodel.edge_param_eid]
        mu_rel_param_es = flownetwork.mu_rel[inversemodel.edge_param_eid]

        return np.pi * np.power(inversemodel.diameter_baselinevalue, 4) / (
                    32. * length_param_es * mu_plasma * mu_rel_param_es) * inversemodel.alpha
