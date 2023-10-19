import sys
from abc import ABC, abstractmethod
import os
import warnings
from collections import defaultdict

import numpy as np
import copy
from types import MappingProxyType


from source.fileio.create_display_plot import s_curve_util, s_curve_personalized_thersholds, util_convergence_plot, s_curve_util_trifurcation, \
    util_convergence_plot_final, \
    percentage_vessel_plot, residual_plot, residual_plot_last_iteration, residual_graph


# from source.bloodflowmodel.flow_balance import FlowBalance


class IterativeRoutine(ABC):
    """
    Abstract base class for the implementations related to the iterative routines.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of IterativeRoutine.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def _iterative_method(self, flownetwork):
        """
        Iterative method used for specific approach
        it may change between implementation the convergence parameter
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    def iterative_function(self, flownetwork):
        """
        Call the functions that solve for the pressures and flow rates.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        self._iterative_method(flownetwork)

    def iterative_routine(self, flownetwork):
        """
        Call the functions that solve for the pressures and flow rates.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.update_transmissibility()
        flownetwork._imp_buildsystem.build_linear_system(flownetwork)
        flownetwork._imp_solver.update_pressure_flow(flownetwork)
        flownetwork._imp_rbcvelocity.update_velocity(flownetwork)
        # inserire flow balance
        flownetwork.check_flow_balance()


class IterativeRoutineNone(IterativeRoutine):
    """
    Class for the single iteration approach
    """

    def _iterative_method(self, flownetwork):
        """
        No iteration are performed
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        pass


class IterativeRoutineMultipleIteration(IterativeRoutine):

    
    def _iterative_method(self, flownetwork):  # , flow_balance):
        """
        """

        # warning handled for np.nan and np.inf
        warnings.filterwarnings("ignore")
        flownetwork.convergence_check = False

        print("Convergence: ...")

        # to reconstruct the position of the value
        # position_array = np.array([i for i in range(flownetwork.nr_of_es)])
        save_data_max_flow, save_data_max_rbc, save_data_max_hemat = [], [], []
        save_data_avg_flow, save_data_avg_rbc, save_data_avg_hemat = [], [], []
        save_data_vessel_flow, save_data_vessel_rbc, save_data_vessel_hemat = [], [], []
        position_prev_future = {}
        # the dict with position e quante volte appare
        # Create a set of all unique positions from both arrays
        all_positions = set(range(len(flownetwork.hd)))
        # Create a dictionary to count how many times each position appears in the top 5
        position_count = defaultdict(int)

        isExist = os.path.exists(self._PARAMETERS['path_output_file'])
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self._PARAMETERS['path_output_file'])

        with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'w') as file:
            file.write(
                f"Network: {self._PARAMETERS['network_name']} \nnr of vs: {flownetwork.nr_of_vs} - nr of boundary vs: {len(flownetwork.boundary_vs)} - nr of es:"
                f" {flownetwork.nr_of_es} \n")

        # with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + "_vessel.txt", 'w') as file:
        #     file.write(f"Network: {self._PARAMETERS['network_name']} \nnr of vs: {flownetwork.nr_of_vs} - nr of boundary vs: {len(flownetwork.boundary_vs)} - nr of es:"
        #                f" {flownetwork.nr_of_es} \n")

        while flownetwork.convergence_check is False:
            # Old hematocrit and flow to be used after in the convergence
            old_hematocrit = copy.deepcopy(flownetwork.hd)
            # old_flow = np.abs(copy.deepcopy(flownetwork.flow_rate))

            # ----- iterative routine -----
            self.iterative_routine(flownetwork)
            flownetwork.iteration += 1

            # ----- iterative routine -----

            # start converging stuff
            # flow_rate = np.abs(copy.deepcopy(flownetwork.flow_rate))
            # hd = copy.deepcopy(flownetwork.hd)
            # node_values = copy.deepcopy(flownetwork.node_values)
            # # Hd
            # cnvg_hem = np.abs(old_hematocrit - hd) / old_hematocrit * 100
            # # new approach
            # cnvg_hem_assolute = np.abs(old_hematocrit - hd)
            # cnvg_hem[cnvg_hem_assolute <= 1e-2] = 0  # 1e-5
            #
            # if flownetwork.iteration < -1:
            #     import matplotlib.pyplot as plt
            #
            #     plt.style.use('seaborn-whitegrid')
            #
            #     # Compute the histogram
            #     hist, bins = np.histogram(cnvg_hem_assolute, bins=100, density=True)
            #
            #     # Calculate bin widths
            #     bin_widths = bins[1:] - bins[:-1]
            #
            #     # Calculate percentages
            #     percentages = hist * bin_widths * 100
            #
            #     # Plot the histogram
            #     plt.figure(figsize=(8, 6), dpi=200)
            #     plt.bar(bins[:-1], percentages, width=bin_widths, align='edge')
            #     plt.xlabel('Absolute hematocrit change')
            #     plt.ylabel('Percentage of vessel')
            #     plt.title('Absolute hematocrit change in non-converging vessel  ')
            #     plt.grid(True)
            #     plt.show()
            #     print()
            #
            # cnvg_hem_avg_per = np.average(cnvg_hem[np.isfinite(cnvg_hem)])
            # cnvg_hem_max_per = np.max(cnvg_hem[np.isfinite(cnvg_hem)])
            #
            # save_data_max_hemat = np.append(save_data_max_hemat, cnvg_hem_max_per)
            # # save_data_max_flow, save_data_max_rbc = np.append(save_data_max_flow, cnvg_flow_max_per), np.append(save_data_max_rbc, cnvg_rbc_max_per)
            # save_data_avg_hemat = np.append(save_data_avg_hemat, cnvg_hem_avg_per)
            # # save_data_avg_flow, save_data_avg_rbc =np.append(save_data_avg_flow, cnvg_flow_avg_per), np.append(save_data_avg_rbc, cnvg_rbc_avg_per)
            # vessel_hd = len(cnvg_hem[cnvg_hem >= 1]) / flownetwork.nr_of_es * 100
            # # vessel_flow, vessel_rbc =len(cnvg_flow[cnvg_flow >= 1]) / flownetwork.nr_of_es * 100, len(cnvg_rbc[cnvg_rbc >= 1]) / flownetwork.nr_of_es * 100
            #
            # save_data_vessel_hemat = np.append(save_data_vessel_hemat, vessel_hd)

            if flownetwork.iteration % 100 == 0 and flownetwork.iteration > 2:
                residual_plot(flownetwork, flownetwork.residualOverIterationMax, flownetwork.residualOverIterationNorm, flownetwork._PARAMETERS, " ", "",
                              "final_convergence")
            if flownetwork.stop:
                flownetwork.convergence_check = True
            else:
                flownetwork.convergence_check = False

            # if flownetwork.iteration == 10000:  # TODO
            #     flownetwork.convergence_check = True
            #
            # elif flownetwork.n_stop == 100:  # TODO: if we want to force it 1 and put back if
            #     flownetwork.convergence_check = True
            #     vessel_value_hd, vessel_value_flow = copy.deepcopy(flownetwork.vessel_value_hd), copy.deepcopy(flownetwork.vessel_value_flow)
            #     node_values_hd, node_values_flow, node_relative_residual = copy.deepcopy(flownetwork.node_values_hd), copy.deepcopy(
            #         flownetwork.node_values_flow), copy.deepcopy(
            #         flownetwork.node_relative_residual)
            #     # print dei vessel hd e flow
            #     for vessel in flownetwork.vessel_general:
            #         residual_graph(flownetwork, vessel_value_hd[vessel], flownetwork._PARAMETERS, vessel, "HD")  # title name
            #         residual_graph(flownetwork, vessel_value_flow[vessel], flownetwork._PARAMETERS, vessel, "Flow")  # title name
            #
            #     for node in flownetwork.node_identifiers:
            #         residual_graph(flownetwork, node_values_hd[node], flownetwork._PARAMETERS, node, "HD_error_for_node")  # title name
            #         residual_graph(flownetwork, node_values_flow[node], flownetwork._PARAMETERS, node, "Flow_error_for_node")  # title name
            #         residual_graph(flownetwork, node_relative_residual[node], flownetwork._PARAMETERS, node, "Residual_Relative")  # title name
            #
            # else:
            #     flownetwork.convergence_check = False

            if flownetwork.iteration == -5:
                with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'a') as file:
                    file.write(f"I:{flownetwork.iteration}")
                    #  and data: HEMATOCRIT {cnvg_hem_avg_per:.5e} {cnvg_hem_max_per:.5e} ALPHA {flownetwork.alpha} VESSEL >1%  hd"
                    # f":{vessel_hd:.5e}% n:{len(cnvg_hem[cnvg_hem >= 1])} \n")

        print(f"Convergence: DONE in -> {flownetwork.iteration} \nAlpha -> {flownetwork.alpha} ")

        # s_curve_util(self._PARAMETERS, flownetwork)
        #
        # s_curve_personalized_thersholds(flownetwork, self._PARAMETERS, 0.1)
        # s_curve_personalized_thersholds(flownetwork, self._PARAMETERS, 0.3)
        # s_curve_personalized_thersholds(flownetwork, self._PARAMETERS, 0.5)
        # s_curve_personalized_thersholds(flownetwork, self._PARAMETERS, 0.7)
        # s_curve_personalized_thersholds(flownetwork, self._PARAMETERS, 0.9)
        #
        # s_curve_util_trifurcation(self._PARAMETERS, flownetwork)
