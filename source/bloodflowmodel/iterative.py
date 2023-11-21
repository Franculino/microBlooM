import pickle
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
    percentage_vessel_plot, residual_plot, residual_plot_last_iteration, residual_graph, frequency_plot


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


def berg_convergence(hd, flow, pressure, hd_current, flow_current):
    residual_part_1 = 1 / (hd_current * flow_current)
    residual_part_2 = hd_current
    residual_parte3 = hd + flow + pressure

    residual_berg = residual_part_1 * residual_part_2 + residual_parte3
    return residual_berg


class IterativeRoutineMultipleIteration(IterativeRoutine):

    def _iterative_method(self, flownetwork):  # , flow_balance):
        """
        """

        # warning handled for np.nan and np.inf
        warnings.filterwarnings("ignore")
        flownetwork.convergence_check = False

        print("Convergence: ...")

        isExist = os.path.exists(self._PARAMETERS['path_output_file'])
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self._PARAMETERS['path_output_file'])

        with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'w') as file:
            file.write(
                f"Network: {self._PARAMETERS['network_name']} \nnr of vs: {flownetwork.nr_of_vs} - nr of boundary vs: {len(flownetwork.boundary_vs)} - nr of es:"
                f" {flownetwork.nr_of_es} \n")

        while flownetwork.convergence_check is False:

            # ----- iterative routine -----
            self.iterative_routine(flownetwork)
            flownetwork.iteration += 1

            if flownetwork.iteration > 3:
                residual_berg = berg_convergence(flownetwork.hd_convergence_criteria_berg, flownetwork.flow_convergence_criteria_berg,
                                                 flownetwork.pressure_convergence_criteria_berg, flownetwork.hd, abs(flownetwork.flow_rate))

            # ----- iterative routine -----

            if flownetwork.iteration % 150 == 0 and flownetwork.iteration > 2:
                residual_plot(flownetwork, flownetwork.residualOverIterationMax, flownetwork.residualOverIterationNorm, flownetwork._PARAMETERS, " ", "",
                              "convergence")

            node_residual, node_relative_residual, local_balance_rbc, node_flow_change_total, indices_over_blue = flownetwork.node_residual, \
                flownetwork.node_relative_residual, \
                flownetwork.local_balance_rbc, flownetwork.node_flow_change_total, flownetwork.indices_over_blue

            if flownetwork.stop and (indices_over_blue is None or len(indices_over_blue) == 0):
                residual_plot(flownetwork, flownetwork.residualOverIterationMax, flownetwork.residualOverIterationNorm, flownetwork._PARAMETERS, " ", "",
                              "convergence")
                f = open('data/out/values_to_print/pckl/' + self._PARAMETERS['network_name'] + '.pckl', 'wb')
                pickle.dump(
                    [flownetwork.flow_rate,
                     flownetwork.node_relative_residual,
                     flownetwork.positions_of_elements_not_in_boundary,
                     flownetwork.node_residual,
                     flownetwork.two_MagnitudeThreshold,
                     flownetwork.node_flow_change,
                     flownetwork.vessel_flow_change,
                     indices_over_blue,
                     node_flow_change_total,
                     flownetwork.vessel_flow_change_total,
                     flownetwork.pressure,
                     flownetwork.hd], f)
                f.close()

            elif flownetwork.stop:  # TODO: if we want to force it 1 and put back if
                flownetwork.convergence_check = True
                residual_plot(flownetwork, flownetwork.residualOverIterationMax, flownetwork.residualOverIterationNorm, flownetwork._PARAMETERS, " ", "",
                              "convergence")

                # IMPORT

                # varibles needed
                # posso eseguire tutte le operazioni perchè è previsto
                if len(node_flow_change_total) != 0:
                    mask = np.ones_like(node_relative_residual, dtype=bool)
                    mask[node_flow_change_total] = False
                    relative_residual_non_converging_without_flow = node_relative_residual[mask]

                    mask2 = np.ones_like(node_residual, dtype=bool)
                    mask2[node_flow_change_total] = False
                    residual_non_converging_without_flow = node_residual[mask2]

                    node_with_flow_change_residual = node_residual[node_flow_change_total]
                    node_with_flow_change_relative_residual = node_relative_residual[node_flow_change_total]

                else:
                    # i need the one wihtout flow changes and i don't have them
                    relative_residual_non_converging_without_flow = node_relative_residual
                    residual_non_converging_without_flow = node_residual

                    # I don't have them
                    node_with_flow_change_residual = np.zeros(len(node_residual))
                    node_with_flow_change_relative_residual = np.zeros(len(node_residual))

                if len(indices_over_blue) != 0:
                    non_convergin_node_residual = node_residual[indices_over_blue]
                    non_convergin_node_relative_residual = node_relative_residual[indices_over_blue]

                else:
                    non_convergin_node_residual = np.zeros(len(node_residual))
                    non_convergin_node_relative_residual = np.zeros(len(node_residual))

                # --- save variables ---
                f = open('data/out/values_to_print/pckl/' + self._PARAMETERS['network_name'] + '.pckl', 'wb')
                pickle.dump(
                    [flownetwork.flow_rate,
                     flownetwork.node_relative_residual,
                     flownetwork.positions_of_elements_not_in_boundary,
                     flownetwork.node_residual,
                     flownetwork.two_MagnitudeThreshold,
                     flownetwork.node_flow_change,
                     flownetwork.vessel_flow_change,
                     indices_over_blue,
                     node_flow_change_total,
                     flownetwork.vessel_flow_change_total,
                     flownetwork.pressure,
                     flownetwork.hd,
                     node_with_flow_change_residual,
                     node_with_flow_change_relative_residual,
                     non_convergin_node_residual,
                     non_convergin_node_relative_residual,
                     relative_residual_non_converging_without_flow,
                     residual_non_converging_without_flow], f)
                f.close()

                with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + "_Gathered_Values.txt", 'w') as file:
                    file.write(f"Network: {self._PARAMETERS['network_name']} \n"
                               f"nr of vs: {flownetwork.nr_of_vs} - nr of boundary vs: {len(flownetwork.boundary_vs)} - nr of es: {flownetwork.nr_of_es} \n"
                               f"\n------- ALL NODES -------\n"
                               f"\nInterval of Residual: \n"
                               f"- max: {max(local_balance_rbc)}\n"
                               f"- mean :{np.mean(local_balance_rbc)}\n"
                               f"- min: {min(local_balance_rbc[local_balance_rbc != 0])} [0 values not printed]\n"

                               f"\nInterval of Relative Residual: \n"
                               f"- max: {max(node_relative_residual)}\n"
                               f"- mean: {np.mean(node_relative_residual)}\n"
                               f"- min: {min(node_relative_residual[node_relative_residual != 0])} [0 values not printed]\n"

                               f"\n------- NON CONVERGING NODES -------\n"
                               f"\nInterval of Residual: \n"
                               f"- max: {max(non_convergin_node_residual)}\n"
                               f"- mean: {np.mean(non_convergin_node_residual)}\n"
                               f"- min: {min(non_convergin_node_residual[non_convergin_node_residual != 0])} [0 values not printed]\n"

                               f"\nInterval of Relative Residual: \n"
                               f"- max: {max(non_convergin_node_relative_residual)}\n"
                               f"- mean: {np.mean(non_convergin_node_relative_residual)}\n"
                               f"- min: {min(non_convergin_node_relative_residual[non_convergin_node_relative_residual != 0])} [0 values not printed]\n")

                    if len(node_flow_change_total) != 0:
                        file.write(f"\n------- NODE WITH FLOW CHANGE BEHAVIOUR -------\n"
                                   f"\nInterval of Residual: \n"
                                   f"- max: {max(node_with_flow_change_residual)}\n"
                                   f"- mean: {np.mean(node_with_flow_change_residual)}\n")
                        try:
                            minimo = min(node_with_flow_change_residual[node_with_flow_change_residual != 0])
                        except ValueError:
                            minimo = 0

                        file.write(f"- min: {minimo} [0 values not printed, if so there are not other values]\n"
                                   f"\nInterval of Relative Residual: \n"
                                   f"- max: {max(node_with_flow_change_relative_residual)}\n"
                                   f"- mean: {np.mean(node_with_flow_change_relative_residual)}\n")

                        try:
                            minimo = min(node_with_flow_change_relative_residual[node_with_flow_change_relative_residual != 0])
                        except ValueError:
                            minimo = 0

                        file.write(f"- min: {minimo} [0 values not printed, if so there are not other values]\n")

                    else:
                        file.write(f"\n------- NODE WITH FLOW CHANGE BEHAVIOUR -------\n"
                                   f"\n NOT present \n")

                    file.write(f"\n------- NON-CONVERGING WITHOUT FLOW DIRECTION CHANGE -------\n"
                               f"\nInterval of Residual: \n"
                               f"- max: {max(residual_non_converging_without_flow)}\n"
                               f"- mean: {np.mean(residual_non_converging_without_flow)}\n")
                    try:
                        minimo = min(residual_non_converging_without_flow[residual_non_converging_without_flow != 0])
                    except ValueError:
                        minimo = 0
                    file.write(f"- min: {minimo} [0 values not printed, if so there are not other values]\n"
                               f"\nInterval of Relative Residual: \n"
                               f"- max: {max(relative_residual_non_converging_without_flow)}\n"
                               f"- mean: {np.mean(relative_residual_non_converging_without_flow)}\n")
                    try:
                        minimo = min(relative_residual_non_converging_without_flow[relative_residual_non_converging_without_flow != 0])
                    except ValueError:
                        minimo = 0
                    file.write(f"- min: {minimo} [0 values not "
                               f"printed, if so there are not other values]\n")
                file.close()
                #
                # # --- ALL NODES ---
                # frequency_plot(flownetwork, node_relative_residual, 'Relative Residual', 'relative residual', 'seagreen', 'auto',
                #                "all_node")
                # frequency_plot(flownetwork, node_relative_residual, 'Relative Residual More Bins', 'relative residual', 'seagreen', 10000,
                #                "all_node")
                # frequency_plot(flownetwork, node_residual, 'Residual', 'residual', 'skyblue', 'auto', "all_node")
                # frequency_plot(flownetwork, node_residual, 'Residual More Bins', 'residual', 'skyblue', 10000, "all_node")
                #
                # # --- NON CONVERGING NODES ---
                # frequency_plot(flownetwork, non_convergin_node_relative_residual, 'Relative Residual',
                #                'relative residual', 'seagreen', 'auto', "non_converging")
                # frequency_plot(flownetwork, non_convergin_node_relative_residual, 'Relative Residual More Bins',
                #                'relative residual', 'seagreen', 10000, "non_converging")
                # frequency_plot(flownetwork, non_convergin_node_residual, 'Residual', 'residual',
                #                'skyblue', 'auto', "non_converging")
                # frequency_plot(flownetwork, non_convergin_node_residual, 'Residual More Bins', 'residual',
                #                'skyblue', 10000, "non_converging")
                #
                # # --- NODE WITH FLOW CHANGE BEHAVIOUR ---
                # if len(node_flow_change_total) != 0:
                #     frequency_plot(flownetwork, node_with_flow_change_relative_residual, 'Relative Residual More Bins',
                #                    'relative residual', 'seagreen', 'auto', "flow_change_total")
                #     frequency_plot(flownetwork, node_with_flow_change_relative_residual, 'Relative Residual More Bins',
                #                    'relative residual', 'seagreen', 10000, "flow_change_total")
                #     frequency_plot(flownetwork, node_with_flow_change_residual, 'Residual', 'residual',
                #                    'skyblue', 'auto', "flow_change_total")
                #     frequency_plot(flownetwork, node_with_flow_change_residual, 'Residual', 'residual',
                #                    'skyblue', 10000, "flow_change_total")
                #
                # # --- NON-CONVERGING WITHOUT FLOW DIRECTION CHANGE ---
                #
                # frequency_plot(flownetwork, relative_residual_non_converging_without_flow, 'Relative Residual', 'relative residual', 'seagreen', 'auto',
                #                "non_converging_without_flow")
                # frequency_plot(flownetwork, relative_residual_non_converging_without_flow, 'Relative Residual More Bins', 'relative residual', 'seagreen',
                #                10000, "non_converging_without_flow")
                #
                # frequency_plot(flownetwork, residual_non_converging_without_flow, 'Residual', 'residual', 'skyblue', 'auto', "non_converging_without_flow")
                # frequency_plot(flownetwork, residual_non_converging_without_flow, 'Residual More Bins', 'residual', 'skyblue', 10000,
                #                "non_converging_without_flow")

            else:
                flownetwork.convergence_check = False

        print(f"Convergence: DONE in -> {flownetwork.iteration} \nAlpha -> {flownetwork.alpha} ")
