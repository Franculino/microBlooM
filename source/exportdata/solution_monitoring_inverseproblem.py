import sys

import numpy as np
import pandas as pd
import igraph
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import source.flow_network as flow_network
import source.inverse_model as inverse_model
from types import MappingProxyType


class SolutionMonitoring(object):
    """
    Class for monitoring the solution for the inverse model.
    """

    def __init__(self, flownetwork: flow_network.FlowNetwork, inversemodel: inverse_model.InverseModel,
                 PARAMETERS: MappingProxyType):

        self.flownetwork = flownetwork
        self.inversemodel = inversemodel
        self._PARAMETERS = PARAMETERS

    def get_arrays_for_plots(self):
        """
        Arrays preparation for plotting the cost function vs iterations. Greate one csv file, containing the value of
        the cost function per each iteration.
        Arrays preparation for plotting the current values of the constraint edge ids as a function of iterations.
        Greate two different csv files, containing the current values of the edge ids which have target measurements
        and the current values of the edge ids which have target ranges.
        """

        csv_path = self._PARAMETERS["csv_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration

        filepath_target = csv_path + "data_convergence_target_" + str(current_iteration) + ".csv"
        filepath_range = csv_path + "data_convergence_range_" + str(current_iteration) + ".csv"
        filepath_cost_function = csv_path + "cost_function_" + str(current_iteration) + ".csv"

        # Create arrays for plotting cost function vs iterations
        f_h = self.inversemodel.f_h
        self.inversemodel.iteration_array = np.append(self.inversemodel.iteration_array, current_iteration)
        self.inversemodel.f_h_array = np.append(self.inversemodel.f_h_array, f_h)

        # Create arrays for plotting the current value - Plot: simulation value vs iterations
        edge_id_target = self.inversemodel.edge_constraint_eid
        edge_constraint_range = self.inversemodel.edge_constraint_range_pm
        edge_constraint_type = self.inversemodel.edge_constraint_type
        edge_constraint_value = self.inversemodel.edge_constraint_value

        flow_rate = self.flownetwork.flow_rate
        rbc_velocity = self.flownetwork.rbc_velocity

        # For the constraint edge with precision measurements, so that the constraint range (edge_tar_range_pm) is zero
        # constraint type: 1: Flow rate constraint & 2: Velocity constraint
        is_target_type_1 = np.logical_and(edge_constraint_type == 1, edge_constraint_range == 0.)
        is_target_type_2 = np.logical_and(edge_constraint_type == 2, edge_constraint_range == 0.)
        current_flow_rate = flow_rate[edge_id_target[is_target_type_1]]
        current_rbc_velocity = rbc_velocity[edge_id_target[is_target_type_2]]
        target_values_flow_rate = edge_constraint_value[is_target_type_1]
        target_values_rbc_velocity = edge_constraint_value[is_target_type_2]

        # Export a csv file for the current values with target value - precision measurements
        data_target = {}
        if np.size(current_flow_rate) > 0 and np.size(current_rbc_velocity) > 0:
            data_target["eid_target"] = np.append(edge_id_target[is_target_type_1], edge_id_target[is_target_type_2])
            data_target["type"] = np.append(np.ones(np.size(current_flow_rate), dtype=int), np.ones(np.size(current_rbc_velocity), dtype=int) * 2)
            data_target["current"] = np.append(current_flow_rate,current_rbc_velocity)
            data_target["target"] = np.append(target_values_flow_rate, target_values_rbc_velocity)
        elif np.size(current_flow_rate) > 0:
            data_target["eid_target"] = edge_id_target[is_target_type_1]
            data_target["type"] = np.ones(np.size(current_flow_rate), dtype=int)
            data_target["current"] = current_flow_rate
            data_target["target"] = target_values_flow_rate
        elif np.size(current_rbc_velocity) > 0:
            data_target["eid_target"] = edge_id_target[is_target_type_2]
            data_target["type"] = np.ones(np.size(current_rbc_velocity), dtype=int) * 2
            data_target["current"] = current_rbc_velocity
            data_target["target"] = target_values_rbc_velocity

        df_target = [pd.DataFrame({k: v}) for k, v in data_target.items()]
        df_target = pd.concat(df_target, axis=1)
        df_target.to_csv(filepath_target, index=False)

        # For the constraint edges with target ranges, so that the constraint range (edge_tar_range_pm) is not zero
        is_range_type_1 = np.logical_and(edge_constraint_type == 1, np.logical_not(edge_constraint_range == 0.))
        is_range_type_2 = np.logical_and(edge_constraint_type == 2, np.logical_not(edge_constraint_range == 0.))
        current_flow_rate_range = flow_rate[edge_id_target[is_range_type_1]]
        current_rbc_velocity_range = rbc_velocity[edge_id_target[is_range_type_2]]
        mean_values_flow_rate = edge_constraint_value[is_range_type_1]
        mean_values_rbc_velocity = edge_constraint_value[is_range_type_2]
        range_values_flow_rate = edge_constraint_range[is_range_type_1]
        range_values_rbc_velocity = edge_constraint_range[is_range_type_2]

        # Export a csv file for the current values with target ranges
        data_range = {}
        if np.size(current_flow_rate_range) > 0 and np.size(current_rbc_velocity_range) > 0:
            data_range["eid_range"] = np.append(edge_id_target[is_range_type_1], edge_id_target[is_range_type_2])
            data_range["type"] = np.append(np.ones(np.size(current_flow_rate_range), dtype=int), np.ones(np.size(current_rbc_velocity_range), dtype=int) * 2)
            data_range["current"] = np.append(current_flow_rate_range, current_rbc_velocity_range)
            data_range["mean"] = np.append(mean_values_flow_rate, mean_values_rbc_velocity)
            data_range["range"] = np.append(range_values_flow_rate,range_values_rbc_velocity)
        elif np.size(current_flow_rate_range) > 0:
            data_range["eid_range"] = edge_id_target[is_range_type_1]
            data_range["type"] = np.ones(np.size(current_flow_rate_range), dtype=int)
            data_range["current"] = current_flow_rate_range
            data_range["mean"] = mean_values_flow_rate
            data_range["range"] = range_values_flow_rate
        elif np.size(current_rbc_velocity_range) > 0:
            data_range["eid_range"] = edge_id_target[is_range_type_2]
            data_range["type"] = np.ones(np.size(current_rbc_velocity_range), dtype=int) * 2
            data_range["current"] = current_rbc_velocity_range
            data_range["mean"] = mean_values_rbc_velocity
            data_range["range"] = range_values_rbc_velocity

        df_range = [pd.DataFrame({k: v}) for k, v in data_range.items()]
        df_range = pd.concat(df_range, axis=1)
        df_range.to_csv(filepath_range, index=False)

        # Export a csv file for cost function vs iterations
        data = {}
        data["iterations"] = self.inversemodel.iteration_array
        data["cost_function"] = self.inversemodel.f_h_array
        df = [pd.DataFrame({k: v}) for k, v in data.items()]
        df = pd.concat(df, axis=1)
        df.to_csv(filepath_cost_function, index=False)

        return

    def plot_cost_fuction_vs_iterations(self):
        """
        Plot the cost function vs iterations.
        """

        png_path = self._PARAMETERS["png_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration

        filepath_png = png_path + "cost_function_vs_iterations_" + str(current_iteration) + ".png"

        f_h_array = self.inversemodel.f_h_array
        iteration_array = self.inversemodel.iteration_array

        # Plot cost function vs iterations
        fig, ax = plt.subplots()
        ax.plot(iteration_array, f_h_array)
        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("Cost function")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(which="major", color="#CCCCCC", linestyle="-")
        ax.grid(which="minor", color="#CCCCCC", linestyle=":")
        fig.savefig(filepath_png, dpi=600)
        plt.close(fig)

        return

    def export_sim_data_node_edge_csv(self):
        """
        Export two different csv files for the simulation values for each node and edge.
        """

        csv_path = self._PARAMETERS["csv_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration

        filepath_vs = csv_path + "data_vs_" + str(current_iteration) + ".csv"
        filepath_es = csv_path + "data_es_" + str(current_iteration) + ".csv"

        # Export a csv file for the vertex attributes
        data_vs = {}
        data_vs["vs_id"] = np.arange(self.flownetwork.nr_of_vs)
        data_vs["coord_x"] = self.flownetwork.xyz[:, 0].transpose()
        data_vs["coord_y"] = self.flownetwork.xyz[:, 1].transpose()
        data_vs["coord_z"] = self.flownetwork.xyz[:, 2].transpose()
        data_vs["pressure"] = self.flownetwork.pressure
        df_vs = [pd.DataFrame({k: v}) for k, v in data_vs.items()]
        df_vs = pd.concat(df_vs, axis=1)
        df_vs.to_csv(filepath_vs, index=False)

        # Export a csv file for the edge attributes
        parameter_baseline_value = np.zeros(self.flownetwork.nr_of_es)
        tuned_parameter = np.zeros(self.flownetwork.nr_of_es)
        labels = [" ", " "]
        if self._PARAMETERS["parameter_space"] == 1:
            parameter_baseline_value = self.inversemodel.diameter_baselinevalue
            tuned_parameter = self.flownetwork.diameter
            labels = ["baseline_diameter", "diameter"]
        elif self._PARAMETERS["parameter_space"] == 2:
            parameter_baseline_value = self.inversemodel.transmiss_baselinevalue
            tuned_parameter = self.flownetwork.transmiss
            labels = ["baseline_transmiss", "transmiss"]

        data_es = {}
        data_es["n1"] = self.flownetwork.edge_list[:, 0]
        data_es["n2"] = self.flownetwork.edge_list[:, 1]
        data_es["es_id"] = np.arange(self.flownetwork.nr_of_es)
        data_es["alpha"] = self.inversemodel.alpha
        data_es[labels[0]] = parameter_baseline_value
        data_es[labels[1]] = tuned_parameter
        data_es["length"] = self.flownetwork.length
        data_es["flow_rate"] = self.flownetwork.flow_rate
        data_es["rbc_velocity"] = self.flownetwork.rbc_velocity
        df_es = [pd.DataFrame({k: v}) for k, v in data_es.items()]
        df_es = pd.concat(df_es, axis=1)
        df_es.to_csv(filepath_es, index=False)

        return

    def export_network_pkl(self):

        pkl_path = self._PARAMETERS["pkl_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration

        filepath = pkl_path + "network_" + str(current_iteration) + ".pkl"

        if self._PARAMETERS["read_network_option"] == 3:
            pkl_path_igraph = self._PARAMETERS["pkl_path_igraph"]
            graph = igraph.Graph.Read_Pickle(pkl_path_igraph)
        else:
            edge_list = self.flownetwork.edge_list
            # Generate igraph based on edge_list
            graph = igraph.Graph(edge_list.tolist())

        if self.flownetwork.diameter is not None:
            graph.es["diameter"] = self.flownetwork.diameter

        if self.flownetwork.length is not None:
            graph.es["length"] = self.flownetwork.length

        if self.flownetwork.flow_rate is not None:
            graph.es["flow_rate"] = self.flownetwork.flow_rate

        if self.flownetwork.rbc_velocity is not None:
            graph.es["rbc_velocity"] = self.flownetwork.rbc_velocity

        if self.flownetwork.ht is not None:
            graph.es["ht"] = self.flownetwork.ht

        graph.vs["xyz"] = self.flownetwork.xyz.tolist()

        if self.flownetwork.pressure is not None:
            graph.vs["pressure"] = self.flownetwork.pressure

        graph.write_pickle(filepath)

        return
