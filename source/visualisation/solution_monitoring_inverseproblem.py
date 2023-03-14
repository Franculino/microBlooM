import numpy as np
import pandas as pd
import igraph
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import source.flow_network as flow_network
import source.inverse_model as inverse_model
from types import MappingProxyType


class SolutionMonitoring(object):

    def __init__(self, flownetwork: flow_network.FlowNetwork, inversemodel: inverse_model.InverseModel,
                 PARAMETERS: MappingProxyType):

        self.flownetwork = flownetwork
        self.inversemodel = inversemodel
        self._PARAMETERS = PARAMETERS

    def get_arrays_for_plots(self):

        current_iteration = self.inversemodel._current_iteration

        # Create arrays for plotting cost_function vs iterations
        f_h = self.inversemodel.f_h
        self.inversemodel._iteration_array = np.append(self.inversemodel._iteration_array, current_iteration)
        self.inversemodel._f_h_array = np.append(self.inversemodel._f_h_array, f_h)

        # Create arrays for plotting tuning value - Plot: tuning_value vs iterations
        edge_id_target = self.inversemodel.edge_constraint_eid
        edge_constraint_range = self.inversemodel.edge_constraint_range_pm
        edge_constraint_type = self.inversemodel.edge_constraint_type

        flow_rate = self.flownetwork.flow_rate
        rbc_velocity = self.flownetwork.rbc_velocity

        # Only for precision measurements, so that the constraint range (edge_tar_range_pm) is zero  - target value
        # constraint type: 1: Flow rate constraint & 2: Velocity constraint
        is_target_type_1 = np.logical_and(edge_constraint_type == 1, edge_constraint_range == 0.)
        is_target_type_2 = np.logical_and(edge_constraint_type == 2, edge_constraint_range == 0.)
        current_flow_rate = flow_rate[edge_id_target[is_target_type_1]]
        current_rbc_velocity = rbc_velocity[edge_id_target[is_target_type_2]]

        if current_iteration == 0:
            self.inversemodel._flow_rate_sim_array = np.array(current_flow_rate)
            self.inversemodel._rbc_velocity_sim_array = np.array(current_rbc_velocity)
        else:
            self.inversemodel._flow_rate_sim_array = np.vstack((self.inversemodel._flow_rate_sim_array,
                                                                   current_flow_rate))
            self.inversemodel._rbc_velocity_sim_array = np.vstack((self.inversemodel._rbc_velocity_sim_array,
                                                                      current_rbc_velocity))

        return

    def plot_cost_fuction_vs_iterations(self):

        png_path = self._PARAMETERS["png_path_solution_monitoring"]
        current_iteration = self.inversemodel._current_iteration

        filepath = png_path + "cost_function_vs_iterations_" + str(current_iteration) + ".png"

        f_h_array = self.inversemodel._f_h_array
        iteration_array = self.inversemodel._iteration_array

        fig, ax = plt.subplots()
        ax.plot(iteration_array, f_h_array)

        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("Cost function")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.grid(which="major", color="#CCCCCC", linestyle="-")
        ax.grid(which="minor", color="#CCCCCC", linestyle=":")

        fig.savefig(filepath, dpi=600)
        plt.close(fig)

        return

    def plot_sim_target_values_vs_iterations(self):

        png_path = self._PARAMETERS["png_path_solution_monitoring"]
        current_iteration = self.inversemodel._current_iteration

        filepath_rbc_velocity = png_path + "tuning_urbc_vs_iterations_" + str(current_iteration) + ".png"
        filepath_flow_rate = png_path + "tuning_flowrate_vs_iterations_" + str(current_iteration) + ".png"

        rbc_velocity_sim = self.inversemodel._rbc_velocity_sim_array
        flow_rate_sim = self.inversemodel._flow_rate_sim_array
        iterations = self.inversemodel._iteration_array

        edge_constraint_value = self.inversemodel.edge_constraint_value
        edge_constraint_range = self.inversemodel.edge_constraint_range_pm
        edge_constraint_type = self.inversemodel.edge_constraint_type
        is_target_type_1 = np.logical_and(edge_constraint_type == 1, edge_constraint_range == 0.)
        is_target_type_2 = np.logical_and(edge_constraint_type == 2, edge_constraint_range == 0.)
        target_values_flow_rate = edge_constraint_value[is_target_type_1]
        target_values_rbc_velocity = edge_constraint_value[is_target_type_2]

        n_edge_rbc_velocity = np.size(rbc_velocity_sim, axis=1)
        n_edge_flow_rate = np.size(flow_rate_sim, axis=1)

        # Plot n_curves per each graph - n_graph is the number of graphs in one panel
        n_curves = 3
        if n_edge_rbc_velocity % n_curves == 0:
            n_graphs_rbc_velocity = n_edge_rbc_velocity // n_curves
        else:
            n_graphs_rbc_velocity = (n_edge_rbc_velocity // n_curves) + 1

        if n_edge_flow_rate % n_curves == 0:
            n_graphs_flow_rate = n_edge_flow_rate // n_curves
        else:
            n_graphs_flow_rate = (n_edge_flow_rate // n_curves) + 1

        n_cols = 3
        if n_graphs_rbc_velocity > 0:
            if n_graphs_rbc_velocity % n_cols == 0:
                n_rows = n_graphs_rbc_velocity // n_cols
            else:
                n_rows = (n_graphs_rbc_velocity // n_cols) + 1

            fig, ax = plt.subplots(n_rows, n_cols, sharex='col', sharey='row')
            n2 = 0
            for j in range(n_cols):
                n_edge_rbc_velocity -= n_curves
                i = 0
                if n_edge_rbc_velocity > 0:
                    n1 = n2
                    n2 += n_curves
                else:
                    n1 = n2
                    n2 = n_edge_rbc_velocity + n_curves + n1
                for k in range(n1, n2):
                    c = ['blue', 'green', 'red']
                    ax[j].plot(iterations, rbc_velocity_sim[:, k], linewidth=2, label='Tuned', color=c[i])
                    ax[j].hlines(y=target_values_rbc_velocity[k], xmin=0, xmax=iterations[-1], linewidth=1,
                                 linestyles='--', label='Target', color=c[i])
                    ax[j].xaxis.set_major_locator(MaxNLocator(integer=True))
                    i += 1
                ax[j].tick_params(axis='x', labelsize=14)
                ax[j].tick_params(axis='y', labelsize=14)
            ax[0].set_ylabel('RBC velocity [m/s]', fontsize=20)
            fig.suptitle('After ' + str(current_iteration) + ' iterations', fontsize=22)
            if n_edge_rbc_velocity % n_curves > 0:
                n_graphs_empty = n_curves - (n_graphs_rbc_velocity % n_curves)
                for i in range(1, n_graphs_empty+1):
                    ax.flat[-i].set_visible(False)  # to remove last empty graphs
            fig.set_size_inches(20, 10, forward=True)
            fig.savefig(filepath_rbc_velocity, dpi=600)
            plt.close(fig)

        n_cols = 3
        if n_graphs_flow_rate > 0:
            if n_graphs_flow_rate % n_cols == 0:
                n_rows = n_graphs_flow_rate // n_cols
            else:
                n_rows = (n_graphs_flow_rate // n_cols) + 1

            fig, ax = plt.subplots(n_rows, n_cols, sharex='col', sharey='row')
            n2 = 0
            for j in range(n_cols):
                n_edge_flow_rate -= n_curves
                i = 0
                if n_edge_flow_rate > 0:
                    n1 = n2
                    n2 += n_curves
                else:
                    n1 = n2
                    n2 = n_edge_flow_rate + n_curves + n1
                for k in range(n1, n2):
                    c = ['blue', 'green', 'red']
                    ax[j].plot(iterations, flow_rate_sim[:, k], linewidth=2, label='Tuned', color=c[i])
                    ax[j].hlines(y=target_values_flow_rate[k], xmin=0, xmax=iterations[-1], linewidth=1,
                                 linestyles='--', label='Target', color=c[i])
                    ax[j].xaxis.set_major_locator(MaxNLocator(integer=True))
                    i += 1
                ax[j].tick_params(axis='x', labelsize=14)
                ax[j].tick_params(axis='y', labelsize=14)
            ax[0].set_ylabel('RBC velocity [m/s]', fontsize=20)
            fig.suptitle('After ' + str(current_iteration) + ' iterations', fontsize=22)
            if n_graphs_flow_rate % n_curves > 0:
                n_graphs_empty = n_curves - (n_graphs_flow_rate % n_curves)
                for i in range(1, n_graphs_empty + 1):
                    ax.flat[-i].set_visible(False)  # to remove last empty graphs
            fig.set_size_inches(20, 10, forward=True)
            fig.savefig(filepath_flow_rate, dpi=600)
            plt.close(fig)

        return

    def export_data_convergence_csv(self):

        csv_path = self._PARAMETERS["csv_path_solution_monitoring"]
        current_iteration = self.inversemodel._current_iteration

        filepath = csv_path + "data_convergence_" + str(current_iteration) + ".csv"

        f_h_array = self.inversemodel._f_h_array
        iteration_array = self.inversemodel._iteration_array

        # Only for precision measurements - target value
        edge_id_target = self.inversemodel.edge_constraint_eid
        edge_constraint_value = self.inversemodel.edge_constraint_value
        edge_constraint_range = self.inversemodel.edge_constraint_range_pm
        edge_constraint_type = self.inversemodel.edge_constraint_type
        tuning_flow_rate = self.inversemodel._flow_rate_sim_array
        tuning_rbc_velocity = self.inversemodel._rbc_velocity_sim_array

        is_target_type_1 = np.logical_and(edge_constraint_type == 1, edge_constraint_range == 0.)
        is_target_type_2 = np.logical_and(edge_constraint_type == 2, edge_constraint_range == 0.)
        edge_id_precision = np.append(edge_id_target[is_target_type_1], edge_id_target[is_target_type_2])
        target_values = np.append(edge_constraint_value[is_target_type_1], edge_constraint_value[is_target_type_2])
        tuning_values = np.append(tuning_flow_rate,tuning_rbc_velocity, axis=1)

        n_iter = np.size(iteration_array)
        target_values_iter = np.vstack([target_values] * n_iter)

        data = {}
        data["iterations"] = iteration_array
        data["cost_function"] = f_h_array
        for i, id in zip (range(np.size(target_values)), edge_id_precision):
            data["target_value_" + str(id)] = target_values_iter[:, i]
            data["tuning_value_" + str(id)] = tuning_values[:, i]

        df = [pd.DataFrame({k: v}) for k, v in data.items()]
        df = pd.concat(df, axis=1)
        df.to_csv(filepath, index=False)

        return

    def export_sim_data_vs_es_csv(self):

        csv_path = self._PARAMETERS["csv_path_solution_monitoring"]
        current_iteration = self.inversemodel._current_iteration

        filepath_vs = csv_path + "data_vs_" + str(current_iteration) + ".csv"
        filepath_es = csv_path + "data_es_" + str(current_iteration) + ".csv"

        # Vertex attributes
        data_vs = {}
        data_vs["vs_id"] = np.arange(self.flownetwork.nr_of_vs)
        data_vs["coord_x"] = self.flownetwork.xyz[:, 0].transpose()
        data_vs["coord_y"] = self.flownetwork.xyz[:, 1].transpose()
        data_vs["coord_z"] = self.flownetwork.xyz[:, 2].transpose()
        data_vs["tuned_pressure"] = self.flownetwork.pressure
        df_vs = [pd.DataFrame({k: v}) for k, v in data_vs.items()]
        df_vs = pd.concat(df_vs, axis=1)
        df_vs.to_csv(filepath_vs, index=False)

        # Edge attributes
        parameter_baseline_value = np.zeros(self.flownetwork.nr_of_es)
        tuned_parameter = np.zeros(self.flownetwork.nr_of_es)
        labels = [" ", " "]
        if self._PARAMETERS["parameter_space"] == 1:
            parameter_baseline_value = self.inversemodel.diameter_baselinevalue
            tuned_parameter = self.flownetwork.diameter
            labels = ["baseline_diameter", "tuned_diameter"]
        elif self._PARAMETERS["parameter_space"] == 2:
            parameter_baseline_value = self.inversemodel.transmiss_baselinevalue
            tuned_parameter = self.flownetwork.transmiss
            labels = ["baseline_transmiss", "tuned_transmiss"]

        data_es = {}
        data_es["es_id"] = np.arange(self.flownetwork.nr_of_es)
        data_es["alpha"] = self.inversemodel.alpha
        data_es[labels[0]] = parameter_baseline_value
        data_es[labels[1]] = tuned_parameter
        data_es["tuned_flow_rate"] = self.flownetwork.flow_rate
        data_es["tuned_rbc_velocity"] = self.flownetwork.rbc_velocity
        df_es = [pd.DataFrame({k: v}) for k, v in data_es.items()]
        df_es = pd.concat(df_es, axis=1)
        df_es.to_csv(filepath_es, index=False)

        return

    def export_network_pkl(self):

        pkl_path = self._PARAMETERS["pkl_path_solution_monitoring"]
        current_iteration = self.inversemodel._current_iteration

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
