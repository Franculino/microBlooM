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
        Arrays preparation for plotting the cost function vs iterations. Create one csv file, containing the value of
        the cost function per each iteration.
        Arrays preparation for plotting the current values of the constraint edge ids as a function of iterations.
        Create two different csv files, containing the current values of the edge ids which have target measurements
        and the current values of the edge ids which have target ranges.
        """

        current_iteration = self.inversemodel.current_iteration

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
        # is_target_type_2 = np.logical_and(edge_constraint_type == 2, edge_constraint_range != 0.00505)
        current_flow_rate = flow_rate[edge_id_target[is_target_type_1]]
        current_rbc_velocity = rbc_velocity[edge_id_target[is_target_type_2]]
        target_values_flow_rate = edge_constraint_value[is_target_type_1]
        target_values_rbc_velocity = edge_constraint_value[is_target_type_2]
        n_target = np.size(current_rbc_velocity) + np.size(current_flow_rate)

        # Export a csv file for the current values with target value - precision measurements
        if np.size(current_flow_rate) > 0 or np.size(current_rbc_velocity) > 0:
            self.inversemodel.evolution_target_values = np.append(self.inversemodel.evolution_target_values,
                                                                current_flow_rate)
            self.inversemodel.evolution_target_values = np.append(self.inversemodel.evolution_target_values,
                                                                current_rbc_velocity)

            self.inversemodel.data_target["eid_target"] = np.append(edge_id_target[is_target_type_1],
                                                                    edge_id_target[is_target_type_2])
            self.inversemodel.data_target["type"] = np.append(np.ones(np.size(current_flow_rate), dtype=int),
                                                              np.ones(np.size(current_rbc_velocity), dtype=int) * 2)
            self.inversemodel.data_target["current"] = self.inversemodel.evolution_target_values
            self.inversemodel.data_target["target"] = np.append(target_values_flow_rate, target_values_rbc_velocity)

        # For the constraint edges with target ranges, so that the constraint range (edge_tar_range_pm) is not zero
        is_range_type_1 = np.logical_and(edge_constraint_type == 1, np.logical_not(edge_constraint_range == 0.))
        is_range_type_2 = np.logical_and(edge_constraint_type == 2, np.logical_not(edge_constraint_range == 0.))
        current_flow_rate_range = flow_rate[edge_id_target[is_range_type_1]]
        current_rbc_velocity_range = rbc_velocity[edge_id_target[is_range_type_2]]
        mean_values_flow_rate = edge_constraint_value[is_range_type_1]
        mean_values_rbc_velocity = edge_constraint_value[is_range_type_2]
        range_values_flow_rate = edge_constraint_range[is_range_type_1]
        range_values_rbc_velocity = edge_constraint_range[is_range_type_2]
        n_range = np.size(current_flow_rate_range) + np.size(current_rbc_velocity_range)

        # Export a csv file for the current values with target ranges
        if np.size(current_flow_rate_range) > 0 or np.size(current_rbc_velocity_range) > 0:
            self.inversemodel.current_range_values = np.append(self.inversemodel.current_range_values,
                                                               current_flow_rate_range)
            self.inversemodel.current_range_values = np.append(self.inversemodel.current_range_values,
                                                               current_rbc_velocity_range)
            self.inversemodel.data_range["eid_range"] = np.append(edge_id_target[is_range_type_1],
                                                                  edge_id_target[is_range_type_2])
            self.inversemodel.data_range["type"] = np.append(np.ones(np.size(current_flow_rate_range), dtype=int),
                                                             np.ones(np.size(current_rbc_velocity_range),
                                                                     dtype=int) * 2)
            self.inversemodel.data_range["current"] = self.inversemodel.current_range_values
            self.inversemodel.data_range["mean"] = np.append(mean_values_flow_rate, mean_values_rbc_velocity)
            self.inversemodel.data_range["range"] = np.append(range_values_flow_rate, range_values_rbc_velocity)

        return n_target, n_range

    def flow_rate_csv(self):
        csv_path = r'C:\Master_thesis_2\BCs_tuning_final_network\Postprocessing\csv_files\flow_rate/flow_rate_'
        current_iteration = self.inversemodel.current_iteration

        filepath_range = (csv_path + "gamma_" + str(self._PARAMETERS["gamma"]) +
                          "_targets_" + str(self._PARAMETERS["n_targets"]) + "_flow_rate_"
                          + str(current_iteration) + ".csv")

        flow_rate = self.flownetwork.flow_rate
        # df_flow_rate = pd.DataFrame({'flow_rate'}, flow_rate)
        df_flow_rate = pd.DataFrame({'flow_rate': pd.Series(flow_rate)})
        df_flow_rate.to_csv(filepath_range, index=False)

        return

    def target_edges_csv(self):
        csv_path = self._PARAMETERS["csv_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration

        filepath_target = (csv_path + "gamma_" + str(self._PARAMETERS["gamma"]) +
                           "_targets_" + str(self._PARAMETERS["n_targets"]) + "_trial_data_convergence_target_" +
                           str(current_iteration) + ".csv")

        df_target = [pd.DataFrame({k: v}) for k, v in self.inversemodel.data_target.items()]
        df_target = pd.concat(df_target, axis=1)
        df_target.to_csv(filepath_target, index=False)

        return

    def pressures_csv(self):
        csv_path = self._PARAMETERS["csv_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration

        filepath_pressures = (csv_path + "gamma_" + str(self._PARAMETERS["gamma"]) +
                           "_targets_" + str(self._PARAMETERS["n_targets"]) + "_pressures_" +
                           str(current_iteration) + ".csv")

        pressure = self.flownetwork.pressure
        df_pressures = pd.DataFrame({'pressures': pd.Series(pressure)})
        df_pressures.to_csv(filepath_pressures, index=False)

        return

    def rbc_velocity_csv(self):
        csv_path = self._PARAMETERS["csv_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration

        filepath_rbc_velocity = (csv_path + "gamma_" + str(self._PARAMETERS["gamma"]) +
                           "_targets_" + str(self._PARAMETERS["n_targets"]) + "_rbc_velocities_" +
                           str(current_iteration) + ".csv")

        rbc_velocities = self.flownetwork.rbc_velocity
        df_rbc_velocity = pd.DataFrame({'rbc_velocities': pd.Series(rbc_velocities)})
        df_rbc_velocity.to_csv(filepath_rbc_velocity, index=False)

        return

    def range_edges_csv(self):
        csv_path = self._PARAMETERS["csv_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration

        filepath_range = (csv_path + "gamma_" + str(self._PARAMETERS["gamma"]) +
                          "_targets_" + str(self._PARAMETERS["n_targets"]) + "_trial_data_convergence_range_" +
                          str(current_iteration) + ".csv")

        df_range = [pd.DataFrame({k: v}) for k, v in self.inversemodel.data_range.items()]
        df_range = pd.concat(df_range, axis=1)
        df_range.to_csv(filepath_range, index=False)

        return

    def target_edges_plot(self, n_target):
        png_path = self._PARAMETERS["png_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration
        filepath_target_png = (png_path + "gamma_" + str(self._PARAMETERS["gamma"]) + "_targets_" +
                               str(self._PARAMETERS["n_targets"]) + "_trial_target_edges_vs_iterations_" +
                               str(current_iteration) + ".png")

        data_target = self.inversemodel.data_target["current"].reshape(-1, n_target)
        iteration_array = self.inversemodel.iteration_array
        # Creates n_target subplots
        fig, axes = plt.subplots(n_target, 1, figsize=(8, 4 * n_target), sharex=True)
        # Iterates over the columns of 'current' and creates a subplot per each one
        for ii in range(n_target):
            ax = axes[ii] if n_target > 1 else axes
            ax.plot(iteration_array, data_target[:, ii], label=f'Edge value evolution')
            ax.axhline(y=self.inversemodel.measurements_value[ii], color='green', linestyle='--'
                       , label='Target value')
            ax.fill_between(iteration_array, self.inversemodel.measurements_value[ii]
                            - self.inversemodel.measurements_value[ii]*self._PARAMETERS["threshold"]
                            , self.inversemodel.measurements_value[ii]
                            + self.inversemodel.measurements_value[ii]*self._PARAMETERS["threshold"]
                            , color='lightgray', alpha=0.5, label='Threshold')
            if self.inversemodel.data_target["type"][ii] == 1:
                ax.set_ylabel(f'Flow rate of the {self.inversemodel.data_target["eid_target"][ii]} edge (ID)')
            else:
                ax.set_ylabel(f'RBC velocity in the {self.inversemodel.data_target["eid_target"][ii]} edge (ID)')
            ax.set_xlabel('Iterations')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()  # Automatically adjusts the space between subplots
        plt.title('Evolution of every target edge value throughout the iterations', fontsize=14)
        fig.savefig(filepath_target_png, dpi=600)
        plt.close(fig)

        return

    def range_edges_plot(self, n_range):
        png_path = self._PARAMETERS["png_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration
        filepath_ranges_png = (png_path + "gamma_" + str(self._PARAMETERS["gamma"])
                               + "_targets_" + str(self._PARAMETERS["n_targets"]) +
                               "_trial_ranges_edges_vs_iterations_" + str(current_iteration) + ".png")

        data_range = self.inversemodel.data_range["current"].reshape(-1, n_range)
        iteration_array = self.inversemodel.iteration_array

        # Creates n_range subplots
        fig_range, axes_range = plt.subplots(n_range, 1, figsize=(8, 4 * n_range), sharex=True)

        # Iterates over the columns of 'current' and creates a subplot per each one
        for jj in range(n_range):
            ax = axes_range[jj] if n_range > 1 else axes_range
            ax.plot(iteration_array, data_range[:, jj], label=f'Edge value evolution')
            ax.fill_between(iteration_array, self.inversemodel.data_range["mean"][jj] -
                            self.inversemodel.data_range["range"][jj], self.inversemodel.data_range["mean"][jj]
                            + self.inversemodel.data_range["range"][jj], color='lightgray', alpha=0.5
                            , label='Range value of the edge')
            if self.inversemodel.data_range["type"][jj] == 1:
                ax.set_ylabel(f'Flow rate of the {self.inversemodel.data_range["eid_range"][jj]} edge (ID)')
            else:
                ax.set_ylabel(f'RBC velocity in the {self.inversemodel.data_range["eid_range"][jj]} edge (ID)')
            ax.set_xlabel('Iterations')
            ax.grid(True)
            ax.legend()

        plt.xlabel('Iterations')

        plt.tight_layout()  # Automatically adjusts the space between subplots
        plt.title('Evolution of every range edge value throughout the iterations', fontsize=14)
        fig_range.savefig(filepath_ranges_png, dpi=600)
        plt.close(fig_range)

        return

    def cost_function_csv(self):
        # Export a csv file for cost function vs iterations
        csv_path = self._PARAMETERS["csv_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration
        filepath_cost_function = (csv_path + "gamma_" + str(self._PARAMETERS["gamma"]) +
                                  "_targets_" + str(self._PARAMETERS["n_targets"]) + "_trial_cost_function_" +
                                  str(current_iteration) + ".csv")

        data = {"iterations": self.inversemodel.iteration_array, "cost_function": self.inversemodel.f_h_array}
        df = [pd.DataFrame({k: v}) for k, v in data.items()]
        df = pd.concat(df, axis=1)
        df.to_csv(filepath_cost_function, index=False)

        return

    def plot_cost_function_vs_iterations(self):
        """
        Plot the cost function vs iterations.
        """

        png_path = self._PARAMETERS["png_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration

        filepath_png = (png_path + "gamma_" + str(self._PARAMETERS["gamma"]) +
                        "_targets_" + str(self._PARAMETERS["n_targets"]) + "_trial_cost_function_vs_iterations_" +
                        str(current_iteration) + ".png")

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
