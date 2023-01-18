from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
from scipy.sparse import coo_matrix
import sys
# todo: this is a construction site


class AdjointMethodParameters(ABC):
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
    def _update_d_f_d_alpha(self, inversemodel, flownetwork):
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
    def _update_d_f_d_pressure(self, inversemodel, flownetwork):
        pass

    @abstractmethod
    def _update_d_g_d_alpha(self, inversemodel, flownetwork):
        pass

    @abstractmethod
    def _get_d_transmiss_d_alpha(self, inversemodel, flownetwork):
        pass

    @abstractmethod
    def _get_d_flowrate_d_alpha(self, inversemodel, flownetwork):
        pass

    @abstractmethod
    def _get_d_velocity_d_alpha(self, inversemodel, flownetwork):
        pass

    def update_partial_derivatives(self, inversemodel, flownetwork):
        self._update_d_f_d_alpha(inversemodel, flownetwork)
        self._update_d_f_d_pressure(inversemodel, flownetwork)
        self._update_d_g_d_alpha(inversemodel, flownetwork)

class AdjointMethodParametersEdge(AdjointMethodParameters):
    """
    Class for a parameter space that is related to edge parameters
    """

    def _update_d_f_d_alpha(self, inversemodel, flownetwork):
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
        inversemodel.d_f_d_alpha *= 0.

        # Compute all partial derivatives with respect to entire parameter vector
        d_flowrate_d_alpha = self._get_d_flowrate_d_alpha(inversemodel, flownetwork)  # d q_ij / d alpha_ij
        d_velocity_d_alpha = self._get_d_velocity_d_alpha(inversemodel, flownetwork)  # d u_ij / d alpha_ij

        # Type of target value (1: Flow rate, 2: Velocity,
        # True if current target value is of the respective type
        is_target_type_1 = np.in1d(inversemodel.edge_constraint_type, 1)  # constraint type 1: Flow rate constraint
        is_target_type_2 = np.in1d(inversemodel.edge_constraint_type, 2)  # constraint type 2: Velocity constraint

        # Identify current simulated value, and corresponding min and max target values
        val_current = np.zeros(inversemodel.nr_of_edge_constraints)
        val_current[is_target_type_1] = flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]]  # constraint type 1: Flow rate constraint
        val_current[is_target_type_2] = flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]]  # constraint type 2: Velocity constraint

        # Identify minimum and maximum target values. If no range is prescribed, target = min = max
        val_min_tar = (inversemodel.edge_constraint_value - inversemodel.edge_constraint_range_pm)
        val_max_tar = (inversemodel.edge_constraint_value + inversemodel.edge_constraint_range_pm)

        # Difference between simulated and target (min, max) value (of all target edges)
        val_difference = np.zeros(inversemodel.nr_of_edge_constraints)
        val_difference[val_current > val_max_tar] = (val_current - val_max_tar)[val_current > val_max_tar]
        val_difference[val_current < val_min_tar] = (val_current - val_min_tar)[val_current < val_min_tar]

        # Update Vector d f / d alpha
        # Constraints of type 1 (flow rate)
        # Find constraint and parameter id for targets of type 1 (flow rate)
        is_current_target = np.logical_and(np.in1d(inversemodel.edge_constraint_eid, inversemodel.edge_param_eid), is_target_type_1)
        is_current_parameter = np.in1d(inversemodel.edge_param_eid, inversemodel.edge_constraint_eid[is_target_type_1])

        q_difference = val_difference[is_current_target]  # q_simulated - q_target/q_min/q_max
        sigma_flow = inversemodel.edge_constraint_sigma[is_current_target]

        # Partial derivative with respect to parameters; all flow rate terms
        inversemodel.d_f_d_alpha[is_current_parameter] = 2. * q_difference / np.square(sigma_flow) * d_flowrate_d_alpha[is_current_parameter]

        # Constraints of type 2 (velocity)
        # Find constraint and parameter id for targets of type 2 (velocity)
        is_current_target = np.logical_and(np.in1d(inversemodel.edge_constraint_eid, inversemodel.edge_param_eid),
                                           is_target_type_2)
        is_current_parameter = np.in1d(inversemodel.edge_param_eid, inversemodel.edge_constraint_eid[is_target_type_2])

        u_difference = val_difference[is_current_target]
        sigma_u = inversemodel.edge_constraint_sigma[is_current_target]

        # Partial derivative with respect to parameters; all velocity terms
        inversemodel.d_f_d_alpha[is_current_parameter] = 2. * u_difference / np.square(sigma_u) * d_velocity_d_alpha[is_current_parameter]

        # Todo: other constraint types (mean flow rate, pressures)

    def _update_d_f_d_pressure(self, inversemodel, flownetwork):

        nr_of_vertices = flownetwork.nr_of_vs
        nr_of_constraints = inversemodel.nr_of_edge_constraints

        # Type of target value (1: Flow rate, 2: Velocity,
        # True if current target value is of the respective type
        is_target_type_1 = np.in1d(inversemodel.edge_constraint_type, 1)  # constraint type 1: Flow rate constraint
        is_target_type_2 = np.in1d(inversemodel.edge_constraint_type, 2)  # constraint type 2: Velocity constraint

        # Identify current simulated value, and corresponding min and max target values
        val_current = np.zeros(nr_of_constraints)
        val_current[is_target_type_1] = flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]]  # constraint type 1: Flow rate constraint
        val_current[is_target_type_2] = flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]]  # constraint type 2: Velocity constraint

        # Identify minimum and maximum target values. If no range is prescribed, target = min = max
        val_min_tar = (inversemodel.edge_constraint_value - inversemodel.edge_constraint_range_pm)
        val_max_tar = (inversemodel.edge_constraint_value + inversemodel.edge_constraint_range_pm)

        # Difference between simulated and target (min, max) value (of all target edges)
        val_difference = np.zeros(inversemodel.nr_of_edge_constraints)
        val_difference[val_current > val_max_tar] = (val_current - val_max_tar)[val_current > val_max_tar]
        val_difference[val_current < val_min_tar] = (val_current - val_min_tar)[val_current < val_min_tar]

        # Bla
        edge_id_target = inversemodel.edge_constraint_eid
        edge_list_target = flownetwork.edge_list[edge_id_target, :]

        col = edge_list_target.reshape(-1)
        row = np.array([[i,i] for i in range(nr_of_constraints)]).reshape(-1)
        data = np.zeros((nr_of_constraints, 2))

        # Constraints of type 1 (flow rate)

        q_difference = val_difference[is_target_type_1]  # q_simulated - q_target/q_min/q_max
        sigma_flow = inversemodel.edge_constraint_sigma[is_target_type_1]

        data[is_target_type_1, 0] = 2. * q_difference / np.square(sigma_flow) * flownetwork.transmiss[edge_id_target[is_target_type_1]]
        data[is_target_type_1, 1] = -data[is_target_type_1, 0]

        # Constraints of type 2 (velocity)

        u_difference = val_difference[is_target_type_2]
        sigma_u = inversemodel.edge_constraint_sigma[is_target_type_2]

        eids_u_target = edge_id_target[is_target_type_2]

        hd_ht_ratio = np.ones(flownetwork.nr_of_es)
        if not 0 in flownetwork.ht:
            hd_ht_ratio = flownetwork.hd / flownetwork.ht

        data[is_target_type_2, 0] = 2. * u_difference / np.square(sigma_u) * 4. * flownetwork.transmiss[eids_u_target] * hd_ht_ratio[eids_u_target] / (np.pi * np.square(flownetwork.diameter[eids_u_target]))
        data[is_target_type_2, 1] = -data[is_target_type_2, 0]
        data = data.reshape(-1)

        df_dp_matrix = coo_matrix((data, (row, col)), shape=(nr_of_constraints, nr_of_vertices))

        inversemodel.d_f_d_pressure = np.array(df_dp_matrix.sum(axis=0)).reshape(-1)

    def _update_d_g_d_alpha(self, inversemodel, flownetwork):

        nr_of_vertices = flownetwork.nr_of_vs
        nr_of_edge_parameters = inversemodel.nr_of_edge_parameters

        eids_params = inversemodel.edge_param_eid
        edge_list_params = flownetwork.edge_list[eids_params, :]

        param_p_diff_ij = flownetwork.pressure[edge_list_params[:, 0]] - flownetwork.pressure[edge_list_params[:, 1]]
        param_p_diff_ji = -param_p_diff_ij

        d_transmiss_d_alpha = self._get_d_transmiss_d_alpha(inversemodel, flownetwork)

        row = np.append(edge_list_params[:, 0], edge_list_params[:, 1])
        col = np.append(np.arange(nr_of_edge_parameters), np.arange(nr_of_edge_parameters))
        data = np.append(param_p_diff_ij * d_transmiss_d_alpha, param_p_diff_ji * d_transmiss_d_alpha)

        inversemodel.d_g_d_alpha = coo_matrix((data, (row, col)), shape=(nr_of_vertices, nr_of_edge_parameters))

    def _get_d_flowrate_d_alpha(self, inversemodel, flownetwork):

        edge_list_params = flownetwork.edge_list[inversemodel.edge_param_eid, :]

        # Return partial derivative d flowrate_ij / d alpha_ij (for all edges with parameters)
        return (flownetwork.pressure[edge_list_params[:, 0]] - flownetwork.pressure[
            edge_list_params[:, 1]]) * self._get_d_transmiss_d_alpha(inversemodel, flownetwork)


class AdjointMethodParametersRelDiam(AdjointMethodParametersEdge):
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

        inversemodel.d_f_d_alpha = np.zeros(inversemodel.nr_of_edge_parameters)
        inversemodel.d_f_d_pressure = np.zeros(flownetwork.nr_of_vs)

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

    def _get_d_velocity_d_alpha(self, inversemodel, flownetwork):

        hd_param_es = flownetwork.hd[inversemodel.edge_param_eid]
        ht_param_es = flownetwork.ht[inversemodel.edge_param_eid]
        diam_param_es = flownetwork.diameter[inversemodel.edge_param_eid]
        flowrate_param_es = flownetwork.flow_rate[inversemodel.edge_param_eid]

        d_flowrate_d_alpha = self._get_d_flowrate_d_alpha(inversemodel, flownetwork)

        hd_ht_ratio = np.ones(np.size(hd_param_es))
        if not 0 in hd_ht_ratio:
            hd_ht_ratio = hd_param_es / ht_param_es

        return 4. / np.pi * hd_ht_ratio * (
                    d_flowrate_d_alpha / np.square(diam_param_es) - 2. * flowrate_param_es / (
                        np.square(inversemodel.diameter_baselinevalue) * np.power(inversemodel.alpha, 3)))

    def _get_d_transmiss_d_alpha(self, inversemodel, flownetwork):

        mu_plasma = self._PARAMETERS["mu_plasma"]
        # Lengths and mu_rel of edges that are parameters
        length_param_es = flownetwork.length[inversemodel.edge_param_eid]
        mu_rel_param_es = flownetwork.mu_rel[inversemodel.edge_param_eid]

        return np.pi * np.power(inversemodel.diameter_baselinevalue, 4) / (
                    32. * length_param_es * mu_plasma * mu_rel_param_es) * inversemodel.alpha


# Todo: Implementations for other edge parameters (abs diameter, rel. transmiss, ...) and vertex parameters (BCs)