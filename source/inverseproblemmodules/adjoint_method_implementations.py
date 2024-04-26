import sys
from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
from scipy.sparse import coo_matrix


class AdjointMethodImplementations(ABC):
    """
    Abstract base class for the implementations specific to the chosen parameter space
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of AdjointMethodParameters class.
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
        Updates the system state of the flow network (such as the diameters, transmissibilities, boundary pressures)
        based on the current parameter value (alpha).
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def _update_d_f_d_alpha(self, inversemodel, flownetwork):
        """
        Computes and updates the partial derivative of the cost function with respect to the parameter.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def _update_d_f_d_alpha_2_3(self, inversemodel, flownetwork):
        """
        Computes and updates the partial derivative of the cost function with respect to the parameter.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    def _update_d_f_d_pressure(self, inversemodel, flownetwork):
        """
        Computes and updates the partial derivative of the cost function 1 with respect to the pressures in
        all vertices.
        Implementation common for all parameters.
        Implementation of Eq. (11.5) in Diss Epp.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        nr_of_vertices = flownetwork.nr_of_vs
        nr_of_constraints = inversemodel.nr_of_edge_constraints

        # Type of target value (1: Flow rate, 2: Velocity,
        # True if current target value is of the respective type
        is_target_type_1 = np.in1d(inversemodel.edge_constraint_type, 1)  # constraint type 1: Flow rate constraint
        is_target_type_2 = np.in1d(inversemodel.edge_constraint_type, 2)  # constraint type 2: Velocity constraint

        # Identify simulated value related to the corresponding target value.
        value_sim = np.zeros(nr_of_constraints)
        # Constraint type 1: Assign current flow rate if it is a constraint type 1.
        value_sim[is_target_type_1] = flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]]
        # Constraint type 2: Assign current velocity if it is a constraint type 2.
        value_sim[is_target_type_2] = flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]]

        # Identify minimum and maximum target values. If no range is prescribed, target = min = max
        # q_tar_min = q_tar - q_range // u_tar_min = u_tar - u_tar_min.
        val_min_tar = (inversemodel.edge_constraint_value - inversemodel.edge_constraint_range_pm)
        val_max_tar = (inversemodel.edge_constraint_value + inversemodel.edge_constraint_range_pm)

        # Difference between simulated and target (min or max) value (of all target edges)
        # val_min < val_sim < val_max:
        val_difference = np.zeros(inversemodel.nr_of_edge_constraints)
        # val_sim > val_max:
        val_difference[value_sim > val_max_tar] = (value_sim - val_max_tar)[value_sim > val_max_tar]
        # val_sim < val_min:
        val_difference[value_sim < val_min_tar] = (value_sim - val_min_tar)[value_sim < val_min_tar]

        # Edge ids of all target edges and corresponding vertex pairs (target vertices). Important: Note that target
        # vertices may belong to multiple target edges, which has to be considered while computing d f/d pressure
        edge_id_target = inversemodel.edge_constraint_eid
        edge_list_target = flownetwork.edge_list[edge_id_target, :]

        # Prepare a sparse matrix which describes the derivative of each individual edge constraint with respect to
        # all pressures (size is nr_of_edge_constraints x nr_of_vertices). To obtain the derivative of the full cost
        # function f with respect to all pressures, this matrix will be summed up over all rows at the end.
        col = edge_list_target.reshape(-1)
        row = np.array([[i, i] for i in range(nr_of_constraints)]).reshape(-1)
        data = np.zeros((nr_of_constraints, 2))

        # Constraints of type 1 (flow rate)
        q_difference = val_difference[is_target_type_1]  # q_simulated - q_target/q_min/q_max
        sigma_flow = inversemodel.edge_constraint_sigma[is_target_type_1]

        # = 2*(q_sim_ij-q_tar)/sigma^2 * T_ij * (+1)
        data[is_target_type_1, 0] = 2. * q_difference / np.square(sigma_flow) * flownetwork.transmiss[
            edge_id_target[is_target_type_1]]
        # = 2*(q_sim_ij-q_tar)/sigma^2 * T_ij * (-1)
        data[is_target_type_1, 1] = -data[is_target_type_1, 0]

        # Constraints of type 2 (velocity)
        u_difference = val_difference[is_target_type_2]  # u_simulated - u_target/u_min/u_max
        sigma_u = inversemodel.edge_constraint_sigma[is_target_type_2]

        # Identify the edge ids of all target edges.
        eids_u_target = edge_id_target[is_target_type_2]

        # Get the derivative of hd with respect to ht of all edges with a target value. Ensure that is always valid and
        # prevent division by 0.
        hd_ht_ratio = np.ones(flownetwork.nr_of_es)
        hd_ht_ratio[flownetwork.ht > 0.] = flownetwork.hd[flownetwork.ht > 0.] / flownetwork.ht[flownetwork.ht > 0.]

        # = 2*(u_sim_ij-u_tar)/sigma^2 * d u_ij / d_pressure_i * (+1)
        data[is_target_type_2, 0] = 2. * u_difference / np.square(sigma_u) * 4. * \
                                    flownetwork.transmiss[eids_u_target] * hd_ht_ratio[eids_u_target] / (
                                            np.pi * np.square(flownetwork.diameter[eids_u_target]))
        # = 2 * (u_sim_ij - u_tar) / sigma ^ 2 * du_ij / d_pressure_j * (-1)
        data[is_target_type_2, 1] = -data[is_target_type_2, 0]

        # Reshape data into 1d array
        data = data.reshape(-1)
        # Generate sparse matrix.
        df_dp_matrix = coo_matrix((data, (row, col)), shape=(nr_of_constraints, nr_of_vertices))
        # Update the vector d f / d pressure
        inversemodel.d_f_d_pressure = np.array(df_dp_matrix.sum(axis=0)).reshape(-1)

    def _update_d_f_d_pressure_2_3(self, inversemodel, flownetwork):
        """
        Computes and updates the partial derivative of the cost function 2 and 3 with respect to the pressures in
        all vertices.
        Implementation common for all parameters.
        Implementation of Eq. (11.5) in Diss Epp.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        weight = self._PARAMETERS["ranges_weight"]

        nr_of_vertices = flownetwork.nr_of_vs
        nr_of_constraints = inversemodel.nr_of_edge_constraints

        # Type of target value (1: Flow rate, 2: Velocity,
        # True if current target value is of the respective type
        is_target_type_1 = np.in1d(inversemodel.edge_constraint_type, 1)  # constraint type 1: Flow rate constraint
        is_target_type_2 = np.in1d(inversemodel.edge_constraint_type, 2)  # constraint type 2: Velocity constraint

        # Target or range constraint
        # True if it is target constraint, False if it is range constraint
        is_target = inversemodel.is_target
        nr_targets = np.count_nonzero(is_target)
        nr_ranges = np.count_nonzero(~is_target)

        # Identify simulated value related to the corresponding target value.
        # Identify simulated value related to the corresponding target value.
        value_sim = np.zeros(inversemodel.nr_of_edge_constraints)  # Initialise with 0. Length is nr of constraints.
        value_sim_abs = np.zeros(inversemodel.nr_of_edge_constraints)
        # Constraint type 1: Assign current flow rate if it is a constraint type 1.
        value_sim[is_target_type_1] = flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]]
        value_sim_abs[is_target_type_1] = np.abs(
            flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]])
        # Constraint type 2: Assign current velocity if it is a constraint type 2.
        value_sim[is_target_type_2] = flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]]
        value_sim_abs[is_target_type_2] = np.abs(
            flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]])

        # Identify minimum and maximum target values. If no range is prescribed, target = min = max
        # q_tar_min = q_tar - q_range // u_tar_min = u_tar - u_tar_min.
        val_min_tar = (inversemodel.edge_constraint_value - inversemodel.edge_constraint_range_pm)
        val_max_tar = (inversemodel.edge_constraint_value + inversemodel.edge_constraint_range_pm)

        # Difference between simulated and target (min or max) value (of all target edges)
        # val_min < val_sim < val_max:
        val_difference = np.zeros(inversemodel.nr_of_edge_constraints)
        # Cuando is_target es True, diferencia sin valor absoluto
        # val_difference[is_target] = value_sim[is_target] - val_max_tar[is_target]
        condition1 = is_target & (value_sim > val_max_tar)
        val_difference[condition1] = (value_sim - val_max_tar)[condition1]
        condition2 = is_target & (value_sim < val_min_tar)
        val_difference[condition2] = (value_sim - val_min_tar)[condition2]
        # Cuando is_target es False y se cumple otra condición, diferencia en valor absoluto
        condition3 = (~is_target) & (value_sim_abs > val_max_tar)
        val_difference[condition3] = (value_sim_abs - val_max_tar)[condition3]
        condition4 = (~is_target) & (value_sim_abs < val_min_tar)
        val_difference[condition4] = (value_sim_abs - val_min_tar)[condition4]

        # Edge ids of all target edges and corresponding vertex pairs (target vertices). Important: Note that target
        # vertices may belong to multiple target edges, which has to be considered while computing d f/d pressure
        edge_id_target = inversemodel.edge_constraint_eid
        edge_list_target = flownetwork.edge_list[edge_id_target, :]

        # Prepare a sparse matrix which describes the derivative of each individual edge constraint with respect to
        # all pressures (size is nr_of_edge_constraints x nr_of_vertices). To obtain the derivative of the full cost
        # function f with respect to all pressures, this matrix will be summed up over all rows at the end.
        col = edge_list_target.reshape(-1)
        row = np.array([[i, i] for i in range(nr_of_constraints)]).reshape(-1)
        data = np.zeros((nr_of_constraints, 2))

        # Constraints of type 1 (flow rate)
        is_target_1 = is_target_type_1 & is_target
        is_range_1 = is_target_type_1 & ~is_target
        q_difference_target = val_difference[is_target_1]  # q_simulated - q_target/q_min/q_max
        q_difference_range = val_difference[is_range_1]
        sigma_flow_target = inversemodel.edge_constraint_sigma[is_target_1]
        sigma_flow_range = inversemodel.edge_constraint_sigma[is_range_1]

        # = 2*(q_sim_ij-q_tar)/sigma^2 * T_ij * (+1). This is not used
        data[is_target_1, 0] = 2. * q_difference_target / np.square(sigma_flow_target) * flownetwork.transmiss[
            edge_id_target[is_target_1]]
        data[is_range_1, 0] = (self._PARAMETERS["ranges_weight"] * 2. * q_difference_range /
                               np.square(sigma_flow_range) * flownetwork.transmiss[edge_id_target[is_range_1]])
        # = 2*(q_sim_ij-q_tar)/sigma^2 * T_ij * (-1)
        data[is_target_1, 1] = -data[is_target_1, 0]
        data[is_range_1, 1] = -data[is_range_1, 0]

        # Constraints of type 2 (velocity)
        is_target_2 = is_target_type_2 & is_target
        is_range_2 = is_target_type_2 & ~is_target
        u_difference_target = val_difference[is_target_2]  # u_simulated - u_target/u_min/u_max
        u_difference_range = val_difference[is_range_2]
        sigma_u_target = inversemodel.edge_constraint_sigma[is_target_2]
        sigma_u_range = inversemodel.edge_constraint_sigma[is_range_2]

        # Identify the edge ids of all target edges.
        eids_u_target = edge_id_target[is_target_2]
        eids_u_range = edge_id_target[is_range_2]

        # Get the derivative of hd with respect to ht of all edges with a target value. Ensure that is always valid and
        # prevent division by 0.
        hd_ht_ratio = np.ones(flownetwork.nr_of_es)
        hd_ht_ratio[flownetwork.ht > 0.] = flownetwork.hd[flownetwork.ht > 0.] / flownetwork.ht[flownetwork.ht > 0.]

        # = 2*(u_sim_ij-u_tar)/sigma^2 * d u_ij / d_pressure_i * (+1)
        data[is_target_2, 0] = (1 - weight) * nr_ranges * 2. * u_difference_target / np.square(sigma_u_target) * 4. * \
                               flownetwork.transmiss[eids_u_target] * hd_ht_ratio[eids_u_target] / (
                                       np.pi * np.square(flownetwork.diameter[eids_u_target]))

        data[is_range_2, 0] = (np.sign(value_sim[is_range_2]) * weight * nr_targets
                               * 2. * u_difference_range / np.square(sigma_u_range) * 4.
                               * flownetwork.transmiss[eids_u_range] * hd_ht_ratio[eids_u_range]
                               / (np.pi * np.square(flownetwork.diameter[eids_u_range])))

        # = 2 * (u_sim_ij - u_tar) / sigma ^ 2 * du_ij / d_pressure_j * (-1)
        data[is_target_2, 1] = -data[is_target_2, 0]
        data[is_range_2, 1] = -data[is_range_2, 0]

        # Reshape data into 1d array
        data = data.reshape(-1)
        # Generate sparse matrix.
        df_dp_matrix = coo_matrix((data, (row, col)), shape=(nr_of_constraints, nr_of_vertices))
        # Update the vector d f / d pressure
        inversemodel.d_f_d_pressure = np.array(df_dp_matrix.sum(axis=0)).reshape(-1)

    @abstractmethod
    def _update_d_g_d_alpha(self, inversemodel, flownetwork):
        """
        Computes and updates the partial derivative of the blood flow model g(p,alpha) with respect to the parameter.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def _get_d_transmiss_d_alpha(self, inversemodel, flownetwork):
        """
        Computes the derivative of the transmissibility with respect to the parameter alpha (d T_ij/d alpha_ij).
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of transmissibility with respect to parameters.
        :rtype: 1d numpy array
        """

    @abstractmethod
    def _get_d_flowrate_d_alpha(self, inversemodel, flownetwork):
        """
        Computes the derivative of the flow rate in all edges with respect to parameters (d q_ij/d alpha_ij).
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of flow rate with respect to parameters.
        :rtype: 1d numpy array
        """

    @abstractmethod
    def _get_d_velocity_d_alpha(self, inversemodel, flownetwork):
        """
        Computes the derivative of the velocity in all edges with respect to parameters (d v_ij/d alpha_ij).
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of velocity with respect to parameters.
        :rtype: 1d numpy array
        """

    def update_partial_derivatives(self, inversemodel, flownetwork):
        """
        Call all methods to update the partial derivatives d f/d alpha, d f/d pressure and d g/d alpha
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        match self._PARAMETERS["cost_function_option"]:
            case 1:
                self._update_d_f_d_pressure(inversemodel, flownetwork)
                self._update_d_f_d_alpha(inversemodel, flownetwork)
            case 2 | 3:
                self._update_d_f_d_pressure_2_3(inversemodel, flownetwork)
                self._update_d_f_d_alpha_2_3(inversemodel, flownetwork)
        self._update_d_g_d_alpha(inversemodel, flownetwork)

    def update_cost_hardconstraint(self, inversemodel, flownetwork):
        """
        Updates the cost function 1 value of the hard constraints
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # Type of target value (1: Flow rate, 2: Velocity)
        # True if current target value is of the respective type
        is_target_type_1 = np.in1d(inversemodel.edge_constraint_type, 1)  # constraint type 1: Flow rate constraint
        is_target_type_2 = np.in1d(inversemodel.edge_constraint_type, 2)  # constraint type 2: Velocity constraint

        # Identify simulated value related to the corresponding target value.
        value_sim = np.zeros(inversemodel.nr_of_edge_constraints)  # Initialise with 0. Length is nr of constraints.
        # Constraint type 1: Assign current flow rate if it is a constraint type 1.
        value_sim[is_target_type_1] = flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]]
        # Constraint type 2: Assign current velocity if it is a constraint type 2.
        value_sim[is_target_type_2] = flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]]

        # Identify minimum and maximum target values. If no range is prescribed, target = min = max
        # q_tar_min = q_tar - q_range // u_tar_min = u_tar - u_tar_min.
        val_min_tar = (inversemodel.edge_constraint_value - inversemodel.edge_constraint_range_pm)
        val_max_tar = (inversemodel.edge_constraint_value + inversemodel.edge_constraint_range_pm)

        edge_constraint_sigma = inversemodel.edge_constraint_sigma

        # Difference between simulated and target (min or max) value (of all target edges)
        # val_min < val_sim < val_max:
        val_difference = np.zeros(inversemodel.nr_of_edge_constraints)
        # val_sim > val_max:
        val_difference[value_sim > val_max_tar] = (value_sim - val_max_tar)[value_sim > val_max_tar]
        # val_sim < val_min:
        val_difference[value_sim < val_min_tar] = (value_sim - val_min_tar)[value_sim < val_min_tar]

        cost_terms = np.square(val_difference / edge_constraint_sigma)

        inversemodel.f_h_target = np.sum(cost_terms)
        inversemodel.f_h = np.sum(cost_terms)

    def update_cost_hardconstraint_2_3(self, inversemodel, flownetwork):
        """
        Updates the cost function 2 or 3 value of the hard constraints
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        weight = self._PARAMETERS["ranges_weight"]

        # Type of target value (1: Flow rate, 2: Velocity)
        # True if current target value is of the respective type
        is_target_type_1 = np.in1d(inversemodel.edge_constraint_type, 1)  # constraint type 1: Flow rate constraint
        is_target_type_2 = np.in1d(inversemodel.edge_constraint_type, 2)  # constraint type 2: Velocity constraint

        # Target or range constraint
        # True if it is target constraint, False if it is range constraint
        is_target = inversemodel.is_target
        nr_targets = np.count_nonzero(is_target)
        nr_ranges = np.count_nonzero(~is_target)

        # Identify simulated value related to the corresponding target value.
        value_sim = np.zeros(inversemodel.nr_of_edge_constraints)  # Initialise with 0. Length is nr of constraints.
        value_sim_abs = np.zeros(inversemodel.nr_of_edge_constraints)
        # Constraint type 1: Assign current flow rate if it is a constraint type 1.
        value_sim[is_target_type_1] = flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]]
        value_sim_abs[is_target_type_1] = np.abs(flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]])
        # Constraint type 2: Assign current velocity if it is a constraint type 2.
        value_sim[is_target_type_2] = flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]]
        value_sim_abs[is_target_type_2] = np.abs(flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]])

        # Identify minimum and maximum target values. If no range is prescribed, target = min = max
        # q_tar_min = q_tar - q_range // u_tar_min = u_tar - u_tar_min.
        val_min_tar = (inversemodel.edge_constraint_value - inversemodel.edge_constraint_range_pm)
        val_max_tar = (inversemodel.edge_constraint_value + inversemodel.edge_constraint_range_pm)

        edge_constraint_sigma = inversemodel.edge_constraint_sigma

        # Difference between simulated and target (min or max) value (of all target edges)
        # val_min < val_sim < val_max:
        val_difference = np.zeros(inversemodel.nr_of_edge_constraints)
        # Cuando is_target es True, diferencia sin valor absoluto
        # val_difference[is_target] = value_sim[is_target] - val_max_tar[is_target]
        condition1 = is_target & (value_sim > val_max_tar)
        val_difference[condition1] = (value_sim - val_max_tar)[condition1]
        condition2 = is_target & (value_sim < val_min_tar)
        val_difference[condition2] = (value_sim - val_min_tar)[condition2]
        # Cuando is_target es False y se cumple otra condición, diferencia en valor absoluto
        condition3 = (~is_target) & (value_sim_abs > val_max_tar)
        val_difference[condition3] = (value_sim_abs - val_max_tar)[condition3]
        condition4 = (~is_target) & (value_sim_abs < val_min_tar)
        val_difference[condition4] = (value_sim_abs - val_min_tar)[condition4]

        # Combine both conditions using the logical OR (|)
        combined_condition = condition3 | condition4 | condition2 | condition1
        # Count the total number of True values in both combined conditions
        inversemodel.vessels_within_range = np.count_nonzero(~combined_condition)

        cost_terms_target = (1 - weight) * np.square(val_difference[is_target]
                                                     / edge_constraint_sigma[is_target]) * nr_ranges  # * nr_boundaries
        cost_terms_range = (weight * np.square(val_difference[~is_target]
                                               / edge_constraint_sigma[~is_target]) * nr_targets)

        inversemodel.f_h_target = np.sum(cost_terms_target)
        inversemodel.f_h_range = np.sum(cost_terms_range)

        inversemodel.f_h = np.sum(cost_terms_target) + np.sum(cost_terms_range)


class AdjointMethodImplementationsEdge(AdjointMethodImplementations, ABC):
    """
    Class for the implementations related to edge-based parameter spaces (diameter, transmissibility)
    """

    def _update_d_f_d_alpha(self, inversemodel, flownetwork):
        """
        Computes and updates the partial derivative of the cost function with respect to the edge parameters (relative
        diameters, absolute diameters, transmissibilities).
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # Initialise derivative d f / d alpha (reset to zero)
        inversemodel.d_f_d_alpha *= 0.

        # Compute the partial derivatives of flow rates and velocities with respect to the entire parameter vector
        # 1d numpy arrays of length "nr of edge parameters"
        d_flowrate_d_alpha = self._get_d_flowrate_d_alpha(inversemodel, flownetwork)  # d q_ij / d alpha_ij
        d_velocity_d_alpha = self._get_d_velocity_d_alpha(inversemodel, flownetwork)  # d u_ij / d alpha_ij

        # Type of target value (1: Flow rate, 2: Velocity, todo: 3-mean flow rate)
        # True if current target value is of the respective type
        is_target_type_1 = np.in1d(inversemodel.edge_constraint_type, 1)  # constraint type 1: Flow rate constraint
        is_target_type_2 = np.in1d(inversemodel.edge_constraint_type, 2)  # constraint type 2: Velocity constraint

        # Identify simulated value related to the corresponding target value.
        value_sim = np.zeros(inversemodel.nr_of_edge_constraints)  # Initialise with 0. Length is nr of constraints.
        # Constraint type 1: Assign current flow rate if it is a constraint type 1.
        value_sim[is_target_type_1] = flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]]
        # Constraint type 2: Assign current velocity if it is a constraint type 2.
        value_sim[is_target_type_2] = flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]]

        # Identify minimum and maximum target values. If no range is prescribed, target = min = max
        # q_tar_min = q_tar - q_range // u_tar_min = u_tar - u_tar_min.
        val_min_tar = (inversemodel.edge_constraint_value - inversemodel.edge_constraint_range_pm)
        val_max_tar = (inversemodel.edge_constraint_value + inversemodel.edge_constraint_range_pm)

        # Difference between simulated and target (min or max) value (of all target edges)
        # val_min < val_sim < val_max:
        val_difference = np.zeros(inversemodel.nr_of_edge_constraints)
        # val_sim > val_max:
        val_difference[value_sim > val_max_tar] = (value_sim - val_max_tar)[value_sim > val_max_tar]
        # val_sim < val_min:
        val_difference[value_sim < val_min_tar] = (value_sim - val_min_tar)[value_sim < val_min_tar]

        # Update the vector d f / d alpha (numpy array of length "nr of edge parameters")

        # Constraints of type 1 (flow rate)
        # True if current edge constraint is of type 1 (flow rate) and has a matching edge parameter.
        is_current_target_q = np.logical_and(np.in1d(inversemodel.edge_constraint_eid, inversemodel.edge_param_eid),
                                           is_target_type_1)
        # True if current edge parameter has a matching edge constraint of type 1 (flow rate)
        is_current_parameter_q = np.in1d(inversemodel.edge_param_eid, inversemodel.edge_constraint_eid[is_target_type_1])
        # is_current_parameter_q array (dimension 35) con False en todos menos en los edges objetivo, que es True

        q_difference = val_difference[is_current_target_q]  # q_simulated - q_target / q_min / q_max
        sigma_flow = inversemodel.edge_constraint_sigma[is_current_target_q]

        # Partial derivative with respect to parameters (all flow rate terms). If a target flow rate has no matching
        # edge parameter, derivative is zero)
        inversemodel.d_f_d_alpha[is_current_parameter_q] = 2. * q_difference / np.square(sigma_flow) * \
                                                           d_flowrate_d_alpha[is_current_parameter_q]

        # Constraints of type 2 (velocity)
        # True if current edge constraint is of type 2 (velocity) and has a matching edge parameter.
        is_current_target_u = np.logical_and(np.in1d(inversemodel.edge_constraint_eid,
                                                     inversemodel.edge_param_eid), is_target_type_2)
        # True if current edge parameter has a matching edge constraint of type 2 (velocity)
        is_current_parameter_u = np.in1d(inversemodel.edge_param_eid,
                                         inversemodel.edge_constraint_eid[is_target_type_2])

        u_difference = val_difference[is_current_target_u]  # u_simulated - u_target / u_min / u_max
        sigma_u = inversemodel.edge_constraint_sigma[is_current_target_u]

        # Partial derivative with respect to parameters (all velocity terms). If a target velocity has no matching
        # edge parameter, derivative is zero
        inversemodel.d_f_d_alpha[is_current_parameter_u] = 2. * u_difference / np.square(sigma_u) * \
                                                           d_velocity_d_alpha[is_current_parameter_u]

    def _update_d_g_d_alpha(self, inversemodel, flownetwork):
        """
        Computes and updates the partial derivative of the blood flow model g(p,alpha) with respect to edge parameters.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        nr_of_vertices = flownetwork.nr_of_vs
        nr_of_edge_parameters = inversemodel.nr_of_edge_parameters  # Number of edges

        # Edge ids and corresponding vertex pairs of all edge parameters.
        eids_params = inversemodel.edge_param_eid
        edge_list_params = flownetwork.edge_list[eids_params, :]

        # Pressure differences over all edges with an edge parameter
        param_p_diff_ij = flownetwork.pressure[edge_list_params[:, 0]] - flownetwork.pressure[edge_list_params[:, 1]]
        param_p_diff_ji = -param_p_diff_ij  # Also consider pressure difference relevant for vertex j

        # Derivative of the transmissibility with respect to the precise parameter
        d_transmiss_d_alpha = self._get_d_transmiss_d_alpha(inversemodel, flownetwork)  # Has length "nr of parameters"

        # Prepare row, col and data vectors to build the sparse matrix
        row = np.append(edge_list_params[:, 0], edge_list_params[:, 1])  # Adds the all [0] and then all [1]
        col = np.append(np.arange(nr_of_edge_parameters), np.arange(nr_of_edge_parameters))  # 0-34 repeated 2times (70)
        data = np.append(param_p_diff_ij * d_transmiss_d_alpha, param_p_diff_ji * d_transmiss_d_alpha)

        # Account for Dirichlet (pressure) boundaries. Set values to 0, if vertex is a pressure boundary.
        pressure_boundary_vs = flownetwork.boundary_vs[flownetwork.boundary_type == 1]  # identify pressure vertices
        is_pressure_boundary_vs = np.in1d(row, pressure_boundary_vs)
        data[is_pressure_boundary_vs] = 0.  # Set value to 0 if current vertex is a pressure boundary vertex

        # Update sparse matrix
        inversemodel.d_g_d_alpha = coo_matrix((data, (row, col)), shape=(nr_of_vertices, nr_of_edge_parameters))
        # print(inversemodel.d_g_d_alpha.todense())

    def _get_d_flowrate_d_alpha(self, inversemodel, flownetwork):
        """
        Computes the derivative of the edge flow rate with respect to the corresponding edge parameter
        (d q_ij/d alpha_ij)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of flow rate with respect to parameters.
        :rtype: 1d numpy array of length "nr of edge parameters"
        """

        # Extract the vertex pairs connected by edges that are included in the parameter space. 2d array of size
        # "nr of edge parameters" x 2
        edge_list_params = flownetwork.edge_list[inversemodel.edge_param_eid, :]

        # Return partial derivative d flowrate_ij / d alpha_ij (for all edges with parameters)
        # d q_ij/d alpha_ij = (p_i-p_j)*d T_ij/d alpha_ij
        return (flownetwork.pressure[edge_list_params[:, 0]] - flownetwork.pressure[
            edge_list_params[:, 1]]) * self._get_d_transmiss_d_alpha(inversemodel, flownetwork)


class AdjointMethodImplementationsRelDiam(AdjointMethodImplementationsEdge):  # Only for parameter_space = 1
    """
    Class for a parameter space that includes all vessel diameters relative to baseline
    """

    def initialise_parameters(self, inversemodel, flownetwork):
        """
        Initialises the parameter space, if relative diameters to baseline are tuned (alpha=d/d_base)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        inversemodel.d_f_d_alpha = np.zeros(inversemodel.nr_of_edge_parameters)
        inversemodel.d_f_d_pressure = np.zeros(flownetwork.nr_of_vs)

        inversemodel.diameter_baselinevalue = flownetwork.diameter[inversemodel.edge_param_eid]  # same length as alpha
        inversemodel.alpha = np.ones(inversemodel.nr_of_edge_parameters)  # Edge parameter initialised with 1
        inversemodel.alpha_prime = np.ones(inversemodel.nr_of_edge_parameters)  # Pseudo edge parameter for range

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
        """
        Computes the derivative of the velocity in all edges with respect to the relative diameter alpha.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of velocity with respect to alpha.
        :rtype: 1d numpy array
        """
        hd_param_es = flownetwork.hd[inversemodel.edge_param_eid]  # read Hd
        ht_param_es = flownetwork.ht[inversemodel.edge_param_eid]  # read Ht
        diam_param_es = flownetwork.diameter[inversemodel.edge_param_eid]  # read diameter of all edges with parameters
        flowrate_param_es = flownetwork.flow_rate[inversemodel.edge_param_eid]  # read flow rate of all edge with param

        # get derivative of flow rate of all edges with a parameter with respect to alpha
        d_flowrate_d_alpha = self._get_d_flowrate_d_alpha(inversemodel, flownetwork)

        # Ensure that ratio hd/ht is always valid (If ht=0 -> set ratio to 1)
        hd_ht_ratio = np.ones(np.size(hd_param_es))
        hd_ht_ratio[ht_param_es > 0.] = hd_param_es[ht_param_es > 0.] / ht_param_es[ht_param_es > 0.]
        # Return the derivative
        return 4. / np.pi * hd_ht_ratio * (
                    d_flowrate_d_alpha / np.square(diam_param_es) - 2. * flowrate_param_es / (
                        np.square(inversemodel.diameter_baselinevalue) * np.power(inversemodel.alpha, 3)))

    def _get_d_transmiss_d_alpha(self, inversemodel, flownetwork):
        """
        Computes the derivative of the transmissibility with respect to the parameter alpha (d T_ij/d alpha_ij).
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of transmissibility with respect to parameters.
        :rtype: 1d numpy array
        """
        mu_plasma = self._PARAMETERS["mu_plasma"]
        # Lengths and mu_rel of edges that are parameters
        length_param_es = flownetwork.length[inversemodel.edge_param_eid]
        mu_rel_param_es = flownetwork.mu_rel[inversemodel.edge_param_eid]

        return np.pi * np.power(inversemodel.diameter_baselinevalue, 4) / (
                    32. * length_param_es * mu_plasma * mu_rel_param_es) * np.power(inversemodel.alpha, 3)

class AdjointMethodImplementationsRelTransmiss(AdjointMethodImplementationsEdge):
    """
    Class for a parameter space that includes all vessel transmissibilities relative to baseline
    """

    def initialise_parameters(self, inversemodel, flownetwork):
        """
        Initialises the parameter space, if relative transmissibilities to baseline are tuned (alpha=T/T_base)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        inversemodel.d_f_d_alpha = np.zeros(inversemodel.nr_of_edge_parameters)
        inversemodel.d_f_d_pressure = np.zeros(flownetwork.nr_of_vs)

        inversemodel.transmiss_baselinevalue = flownetwork.transmiss[inversemodel.edge_param_eid]  # same len as alpha
        inversemodel.alpha = np.ones(inversemodel.nr_of_edge_parameters)  # Edge parameter initialised with 1
        inversemodel.alpha_prime = np.ones(inversemodel.nr_of_edge_parameters)  # Pseudo edge parameter for range

    def update_state(self, inversemodel, flownetwork):
        """
        Updates the system state of the flow network (here: transmissibility) based on the
        current parameter value (here: alpha = T/T_base)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.transmiss[inversemodel.edge_param_eid] = inversemodel.transmiss_baselinevalue * inversemodel.alpha

        # Update the diameter of the vascular network based on the tuned transmissibility. Make diameter consistent with
        # transmissibility again. Warning: mu_rel based on d from previous iteration step is used.
        mu_plasma = self._PARAMETERS["mu_plasma"]
        flownetwork.diameter = np.power(128. / np.pi * flownetwork.transmiss * flownetwork.length *
                                        mu_plasma * flownetwork.mu_rel, .25)

    def _get_d_velocity_d_alpha(self, inversemodel, flownetwork):
        """
        Computes the derivative of the velocity in all edges with respect to the relative diameter alpha.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of velocity with respect to alpha.
        :rtype: 1d numpy array
        """
        hd_param_es = flownetwork.hd[inversemodel.edge_param_eid]  # read Hd
        ht_param_es = flownetwork.ht[inversemodel.edge_param_eid]  # read Ht

        # read transmissibility of all edges with parameters
        transmiss_param_es = flownetwork.transmiss[inversemodel.edge_param_eid]

        mu_plasma = self._PARAMETERS["mu_plasma"]
        # Lengths and mu_rel of edges that are parameters
        length_param_es = flownetwork.length[inversemodel.edge_param_eid]
        mu_rel_param_es = flownetwork.mu_rel[inversemodel.edge_param_eid]

        edge_list_params = flownetwork.edge_list[inversemodel.edge_param_eid, :]

        # Ensure that ratio hd/ht is always valid (If ht=0 -> set ratio to 1)
        hd_ht_ratio = np.ones(np.size(hd_param_es))
        hd_ht_ratio[ht_param_es > 0.] = hd_param_es[ht_param_es > 0.] / ht_param_es[ht_param_es > 0.]

        return (flownetwork.pressure[edge_list_params[:, 0]] - flownetwork.pressure[edge_list_params[:, 1]]) * \
            hd_ht_ratio / np.sqrt(32 * np.pi * length_param_es * mu_plasma * mu_rel_param_es) * \
            np.power(transmiss_param_es, -.5) * inversemodel.transmiss_baselinevalue

    def _get_d_transmiss_d_alpha(self, inversemodel, flownetwork):
        """
        Computes the derivative of the transmissibility with respect to the parameter alpha (d T_ij/d alpha_ij).
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of transmissibility with respect to parameters.
        :rtype: 1d numpy array
        """

        return inversemodel.transmiss_baselinevalue


class AdjointMethodImplementationsVertex(AdjointMethodImplementations, ABC):
    """
    Abstract class for the implementations related to vertex-based parameter spaces (e.g. boundary conditions)
    """


class AdjointMethodImplementationsAbsBoundaryPressure(AdjointMethodImplementationsVertex):
    """
    Class for a parameter space that includes absolute boundary pressure condition values
    """

    def _update_d_f_d_alpha(self, inversemodel, flownetwork):
        """
        Computes and updates the partial derivative of the cost function with respect to the vertex parameters
        (boundary pressure).
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # Type of target value (1: Flow rate, 2: Velocity,
        # True if current target value is of the respective type
        is_target_type_1 = np.in1d(inversemodel.edge_constraint_type, 1)  # constraint type 1: Flow rate constraint
        is_target_type_2 = np.in1d(inversemodel.edge_constraint_type, 2)  # constraint type 2: Velocity constraint

        is_target = inversemodel.is_target

        # Identify simulated value related to the corresponding target value.
        # Identify simulated value related to the corresponding target value.
        value_sim = np.zeros(inversemodel.nr_of_edge_constraints)  # Initialise with 0. Length is nr of constraints.
        value_sim_abs = np.zeros(inversemodel.nr_of_edge_constraints)
        # Constraint type 1: Assign current flow rate if it is a constraint type 1.
        value_sim[is_target_type_1] = flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]]
        value_sim_abs[is_target_type_1] = np.abs(
            flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]])
        # Constraint type 2: Assign current velocity if it is a constraint type 2.
        value_sim[is_target_type_2] = flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]]
        value_sim_abs[is_target_type_2] = np.abs(
            flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]])

        # Identify minimum and maximum target values. If no range is prescribed, target = min = max
        # q_tar_min = q_tar - q_range // u_tar_min = u_tar - u_tar_min.
        val_min_tar = (inversemodel.edge_constraint_value - inversemodel.edge_constraint_range_pm)
        val_max_tar = (inversemodel.edge_constraint_value + inversemodel.edge_constraint_range_pm)

        # Difference between simulated and target (min or max) value (of all target edges)
        # val_min < val_sim < val_max:
        val_difference = np.zeros(inversemodel.nr_of_edge_constraints)
        # Cuando is_target es True, diferencia sin valor absoluto
        # val_difference[is_target] = value_sim[is_target] - val_max_tar[is_target]
        condition1 = is_target & (value_sim > val_max_tar)
        val_difference[condition1] = (value_sim - val_max_tar)[condition1]
        condition2 = is_target & (value_sim < val_min_tar)
        val_difference[condition2] = (value_sim - val_min_tar)[condition2]
        # Cuando is_target es False y se cumple otra condición, diferencia en valor absoluto
        condition3 = (~is_target) & (value_sim_abs > val_max_tar)
        val_difference[condition3] = (value_sim_abs - val_max_tar)[condition3]
        condition4 = (~is_target) & (value_sim_abs < val_min_tar)
        val_difference[condition4] = (value_sim_abs - val_min_tar)[condition4]

        # Identify the edge ids of all target edges in the boundary and in the parameter space
        eids_u_target_boundary = inversemodel.targets_boundary_ed
        # Get node pairs for edges in eids_u_target_boundary
        edge_pairs = flownetwork.edge_list[eids_u_target_boundary]
        # Check if both nodes are not in inversemodel.vertex_param_vid
        mask = ~np.any(np.isin(edge_pairs, inversemodel.vertex_param_vid), axis=1)
        # Apply the mask to remove edge ids where none of the nodes are in vertex_param_vid
        eids_u_target_boundary_param_space = eids_u_target_boundary[~mask]
        position_targets_contraints = np.searchsorted(inversemodel.edge_constraint_eid,
                                                      eids_u_target_boundary_param_space)

        # Edge ids of all target edges and corresponding vertex pairs (target vertices). Important: Note that target
        # vertices may belong to multiple target edges, which has to be considered while computing d f/d pressure
        # Constraints of type 2 (velocity)
        u_difference_target_boundary = val_difference[position_targets_contraints]  # u_simulated - u_target/u_min/u_max
        sigma_u_target_boundary = inversemodel.edge_constraint_sigma[position_targets_contraints]

        # Get the derivative of hd with respect to ht of all edges with a target value. Ensure that is always valid and
        # prevent division by 0.
        hd_ht_ratio = np.ones(flownetwork.nr_of_es)
        hd_ht_ratio[flownetwork.ht > 0.] = flownetwork.hd[flownetwork.ht > 0.] / flownetwork.ht[flownetwork.ht > 0.]

        d_u_d_alpha_target, d_u_d_alpha_range = self._get_d_velocity_d_alpha(inversemodel, flownetwork)

        df_dalpha_targets = (2. * u_difference_target_boundary / np.square(sigma_u_target_boundary)
                             * d_u_d_alpha_target)

        # Create array size flownetwork.boundary_vs
        target_array = np.zeros(inversemodel.nr_of_parameters)
        position_target = np.where(np.isin(inversemodel.vertex_param_vid, inversemodel.targets_boundary_v))[0]
        neg_index = np.where(np.isin(inversemodel.vertex_param_vid, flownetwork.edge_list[:, 1]))[0]

        # Asigna los valores de df_dalpha_targets a los índices correspondientes en target_array
        target_array[position_target] = df_dalpha_targets
        target_ranges_array = target_array
        target_ranges_array[neg_index] *= -1
        # Initialise derivative d f / d alpha (reset to zero)
        inversemodel.d_f_d_alpha = target_ranges_array  # + pressure_change_term

    def _update_d_f_d_alpha_2_3(self, inversemodel, flownetwork):
        """
        Computes and updates the partial derivative of the cost function 2 or 3with respect to the vertex parameters
        (boundary pressure).
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        weight = self._PARAMETERS["ranges_weight"]
        # Type of target value (1: Flow rate, 2: Velocity,
        # True if current target value is of the respective type
        is_target_type_1 = np.in1d(inversemodel.edge_constraint_type, 1)  # constraint type 1: Flow rate constraint
        is_target_type_2 = np.in1d(inversemodel.edge_constraint_type, 2)  # constraint type 2: Velocity constraint

        is_target = inversemodel.is_target
        nr_targets = np.count_nonzero(is_target)
        nr_ranges = np.count_nonzero(~is_target)

        # Identify simulated value related to the corresponding target value.
        # Identify simulated value related to the corresponding target value.
        value_sim = np.zeros(inversemodel.nr_of_edge_constraints)  # Initialise with 0. Length is nr of constraints.
        value_sim_abs = np.zeros(inversemodel.nr_of_edge_constraints)
        # Constraint type 1: Assign current flow rate if it is a constraint type 1.
        value_sim[is_target_type_1] = flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]]
        value_sim_abs[is_target_type_1] = np.abs(
            flownetwork.flow_rate[inversemodel.edge_constraint_eid[is_target_type_1]])
        # Constraint type 2: Assign current velocity if it is a constraint type 2.
        value_sim[is_target_type_2] = flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]]
        value_sim_abs[is_target_type_2] = np.abs(
            flownetwork.rbc_velocity[inversemodel.edge_constraint_eid[is_target_type_2]])

        # Identify minimum and maximum target values. If no range is prescribed, target = min = max
        # q_tar_min = q_tar - q_range // u_tar_min = u_tar - u_tar_min.
        val_min_tar = (inversemodel.edge_constraint_value - inversemodel.edge_constraint_range_pm)
        val_max_tar = (inversemodel.edge_constraint_value + inversemodel.edge_constraint_range_pm)

        # Difference between simulated and target (min or max) value (of all target edges)
        # val_min < val_sim < val_max:
        val_difference = np.zeros(inversemodel.nr_of_edge_constraints)
        # Cuando is_target es True, diferencia sin valor absoluto
        # val_difference[is_target] = value_sim[is_target] - val_max_tar[is_target]
        condition1 = is_target & (value_sim > val_max_tar)
        val_difference[condition1] = (value_sim - val_max_tar)[condition1]
        condition2 = is_target & (value_sim < val_min_tar)
        val_difference[condition2] = (value_sim - val_min_tar)[condition2]
        # Cuando is_target es False y se cumple otra condición, diferencia en valor absoluto
        condition3 = (~is_target) & (value_sim_abs > val_max_tar)
        val_difference[condition3] = (value_sim_abs - val_max_tar)[condition3]
        condition4 = (~is_target) & (value_sim_abs < val_min_tar)
        val_difference[condition4] = (value_sim_abs - val_min_tar)[condition4]

        # Identify the edge ids of all target edges in the boundary and in the parameter space
        eids_u_target_boundary = inversemodel.targets_boundary_ed
        # Get node pairs for edges in eids_u_target_boundary
        edge_pairs = flownetwork.edge_list[eids_u_target_boundary]
        # Check if both nodes are not in inversemodel.vertex_param_vid
        mask = ~np.any(np.isin(edge_pairs, inversemodel.vertex_param_vid), axis=1)
        # Apply the mask to remove edge ids where none of the nodes are in vertex_param_vid
        eids_u_target_boundary_param_space = eids_u_target_boundary[~mask]
        position_targets_contraints = np.searchsorted(inversemodel.edge_constraint_eid, eids_u_target_boundary_param_space)

        eids_u_range_boundary = inversemodel.ranges_boundary_ed
        # Get node pairs for edges in eids_u_target_boundary
        edge_pairs = flownetwork.edge_list[eids_u_range_boundary]
        # Check if both nodes are not in inversemodel.vertex_param_vid
        mask = ~np.any(np.isin(edge_pairs, inversemodel.vertex_param_vid), axis=1)
        # Apply the mask to remove edge ids where none of the nodes are in vertex_param_vid
        eids_u_ranges_boundary_param_space = eids_u_range_boundary[~mask]
        position_ranges_contraints = np.searchsorted(inversemodel.edge_constraint_eid,
                                                     eids_u_ranges_boundary_param_space)

        # Edge ids of all target edges and corresponding vertex pairs (target vertices). Important: Note that target
        # vertices may belong to multiple target edges, which has to be considered while computing d f/d pressure
        edge_id_target = inversemodel.edge_constraint_eid
        # Constraints of type 2 (velocity)
        u_difference_target_boundary = val_difference[position_targets_contraints]  # u_simulated - u_target/u_min/u_max
        u_difference_range_boundary = val_difference[position_ranges_contraints]
        sigma_u_target_boundary = inversemodel.edge_constraint_sigma[position_targets_contraints]
        sigma_u_range_boundary = inversemodel.edge_constraint_sigma[position_ranges_contraints]

        # Get the derivative of hd with respect to ht of all edges with a target value. Ensure that is always valid and
        # prevent division by 0.
        hd_ht_ratio = np.ones(flownetwork.nr_of_es)
        hd_ht_ratio[flownetwork.ht > 0.] = flownetwork.hd[flownetwork.ht > 0.] / flownetwork.ht[flownetwork.ht > 0.]

        d_u_d_alpha_target, d_u_d_alpha_range = self._get_d_velocity_d_alpha(inversemodel, flownetwork)

        df_dalpha_targets = ((1 - weight) * 2. * u_difference_target_boundary * nr_ranges /
                             np.square(sigma_u_target_boundary) * d_u_d_alpha_target)

        df_dalpha_ranges = (np.sign(value_sim[position_ranges_contraints]) * weight * nr_targets * 2. *
                            u_difference_range_boundary / np.square(sigma_u_range_boundary) * d_u_d_alpha_range)

        # Create array size flownetwork.boundary_vs
        target_array = np.zeros(inversemodel.nr_of_parameters)
        ranges_array = np.zeros(inversemodel.nr_of_parameters)
        position_target = np.where(np.isin(inversemodel.vertex_param_vid, inversemodel.targets_boundary_v))[0]
        position_ranges = np.where(np.isin(inversemodel.vertex_param_vid, inversemodel.ranges_boundary_v))[0]
        neg_index = np.where(np.isin(inversemodel.vertex_param_vid, flownetwork.edge_list[:, 1]))[0]

        # Asigna los valores de df_dalpha_targets a los índices correspondientes en target_array
        target_array[position_target] = df_dalpha_targets
        ranges_array[position_ranges] = df_dalpha_ranges
        target_ranges_array = target_array + ranges_array
        target_ranges_array[neg_index] *= -1
        # Initialise derivative d f / d alpha (reset to zero)
        inversemodel.d_f_d_alpha = target_ranges_array  # + pressure_change_term

    def _update_d_g_d_alpha(self, inversemodel, flownetwork):
        """
        Computes and updates the partial derivative of the blood flow model g(p,alpha) with respect to vertex
        parameters (boundary pressures).
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        nr_of_vertices = flownetwork.nr_of_vs
        nr_of_vertex_parameters = inversemodel.nr_of_vertex_parameters # Number of boundaries where the pressure is tuned

        vids_params = inversemodel.vertex_param_vid  # Indices of vertices with a parameter
        param_ids = np.arange(nr_of_vertex_parameters)  # All parameter indices (0 ... nr_of_vertex_parameters-1)

        # Build sparse matrix
        row = vids_params
        col = param_ids
        data = -np.ones(nr_of_vertex_parameters)

        # Update matrix
        inversemodel.d_g_d_alpha = coo_matrix((data, (row, col)), shape=(nr_of_vertices, nr_of_vertex_parameters))

    def _get_d_flowrate_d_alpha(self, inversemodel, flownetwork):
        """
        Computes the derivative of the edge flow rate with respect to the corresponding vertex parameter (boundary
        pressure)
        (d q_ij/d p_0)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of flow rate with respect to parameters.
        :rtype: 1d numpy array of length "nr of edge parameters"
        """
        return np.zeros(inversemodel.nr_of_parameters)

    def initialise_parameters(self, inversemodel, flownetwork):
        """
        Initialises the parameter space, if boundary pressures are tuned (alpha=p_0)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        inversemodel.d_f_d_alpha = np.zeros(inversemodel.nr_of_vertex_parameters)
        inversemodel.d_f_d_pressure = np.zeros(flownetwork.nr_of_vs)

        inversemodel.diameter_baselinevalue = None

        # identify the boundary vertices which are parameters
        is_boundary_parameter = np.in1d(flownetwork.boundary_vs, inversemodel.vertex_param_vid)

        # Vertex parameter and pseudo parameter initialised with base value
        inversemodel.alpha = flownetwork.boundary_val[is_boundary_parameter]
        inversemodel.alpha_prime = flownetwork.boundary_val[is_boundary_parameter]
        inversemodel.boundary_pressure_baseline = flownetwork.boundary_val[is_boundary_parameter]
        inversemodel.boundary_pressure_previous = flownetwork.boundary_val[is_boundary_parameter]

    def update_state(self, inversemodel, flownetwork):
        """
        Updates the system state of the flow network (here: absolute boundary pressure) based on the
        current parameter value (here: alpha = p_0)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        is_boundary_parameter = np.in1d(flownetwork.boundary_vs, inversemodel.vertex_param_vid)
        inversemodel.boundary_pressure_previous = flownetwork.boundary_val[is_boundary_parameter]

        flownetwork.boundary_val[is_boundary_parameter] = inversemodel.alpha

    def _get_d_velocity_d_alpha(self, inversemodel, flownetwork):
        """
        Computes the derivative of the edge velocity with respect to the corresponding vertex parameter (boundary
        pressure)
        (d v_ij/d p_0)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of velocity with respect to alpha.
        :rtype: 1d numpy array
        """
        # Identify the edge ids of all target edges in the boundary and in the parameter space
        eids_u_target_boundary = inversemodel.targets_boundary_ed
        # Get node pairs for edges in eids_u_target_boundary
        edge_pairs = flownetwork.edge_list[eids_u_target_boundary]
        # Check if both nodes are not in inversemodel.vertex_param_vid
        mask = ~np.any(np.isin(edge_pairs, inversemodel.vertex_param_vid), axis=1)
        # Apply the mask to remove edge ids where none of the nodes are in vertex_param_vid
        eids_u_target_boundary_param_space = eids_u_target_boundary[~mask]
        position_targets_contraints = np.searchsorted(inversemodel.edge_constraint_eid,
                                                      eids_u_target_boundary_param_space)

        eids_u_range_boundary = inversemodel.ranges_boundary_ed
        # Get node pairs for edges in eids_u_target_boundary
        edge_pairs = flownetwork.edge_list[eids_u_range_boundary]
        # Check if both nodes are not in inversemodel.vertex_param_vid
        mask = ~np.any(np.isin(edge_pairs, inversemodel.vertex_param_vid), axis=1)
        # Apply the mask to remove edge ids where none of the nodes are in vertex_param_vid
        eids_u_ranges_boundary_param_space = eids_u_range_boundary[~mask]
        position_ranges_contraints = np.searchsorted(inversemodel.edge_constraint_eid,
                                                     eids_u_ranges_boundary_param_space)

        # Get the derivative of hd with respect to ht of all edges with a target value. Ensure that is always valid and
        # prevent division by 0.
        hd_ht_ratio = np.ones(flownetwork.nr_of_es)
        hd_ht_ratio[flownetwork.ht > 0.] = flownetwork.hd[flownetwork.ht > 0.] / flownetwork.ht[flownetwork.ht > 0.]

        d_u_d_alpha_target = (4. * flownetwork.transmiss[eids_u_target_boundary_param_space]
                     * hd_ht_ratio[eids_u_target_boundary_param_space] / (np.pi * np.square(
                    flownetwork.diameter[eids_u_target_boundary_param_space])))

        d_u_d_alpha_range = (4. * flownetwork.transmiss[eids_u_ranges_boundary_param_space]
                            * hd_ht_ratio[eids_u_ranges_boundary_param_space]
                            / (np.pi * np.square(flownetwork.diameter[eids_u_ranges_boundary_param_space])))

        return d_u_d_alpha_target, d_u_d_alpha_range

    def _get_d_transmiss_d_alpha(self, inversemodel, flownetwork):
        """
        Computes the derivative of the edge transmissibility with respect to the corresponding vertex parameter
        (boundary pressure) (d T_ij/d p_0)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Derivative of transmissibility with respect to parameters.
        :rtype: 1d numpy array
        """
        return np.zeros(inversemodel.nr_of_parameters)