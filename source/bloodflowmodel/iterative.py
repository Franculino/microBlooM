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
    percentage_vessel_plot, residual_plot, residual_plot_last_iteration, residual_graph, frequency_plot, residual_plot_berg, residual_plot_rasmussen, \
    residual_plot_berg_subset


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


def berg_convergence(self, flownetwork):
    # 1/ (h_d Q)^n _i
    # interpret as the inlow RBcs at iteration n
    residual_part_1 = 1 / (self._PARAMETERS["boundary_hematocrit"] * flownetwork.inflow)
    flownetwork.Berg1.append(residual_part_1)

    # sum(|delta(H_dQ)|^n_k) interpret as the leakage of red blood cells at inner vertex k
    residual_part_2 = sum(abs(flownetwork.local_balance_rbc))
    flownetwork.Berg2.append(residual_part_2)

    residual12 = residual_part_1 * residual_part_2
    flownetwork.BergFirstPartEq.append(residual12)

    # ||X^n - X^n-1|| / X^n_i
    residual_parte_3 = flownetwork.pressure_convergence_criteria_berg + flownetwork.flow_convergence_criteria_berg + flownetwork.hd_convergence_criteria_berg
    flownetwork.BergPressure.append(flownetwork.pressure_convergence_criteria_berg)
    flownetwork.BergFlow.append(flownetwork.flow_convergence_criteria_berg)
    flownetwork.BergHD.append(flownetwork.hd_convergence_criteria_berg)
    flownetwork.BergSecondPartEq.append(residual_parte_3)

    # 1/ (h_d Q)^n _i * sum(|delta(H_dQ)|^n_k) + ||X^n - X^n-1|| / X^n_i
    residual = residual12 + residual_parte_3
    flownetwork.bergIteration.append(residual)
    return residual


class IterativeRoutineMultipleIteration(IterativeRoutine):

    def _iterative_method(self, flownetwork):  # , flow_balance):

        # warning handled for np.nan and np.inf
        warnings.filterwarnings("ignore")
        flownetwork.convergence_check = False

        print("Convergence: ...")

        # isExist = os.path.exists(self._PARAMETERS['path_output_file'])
        # if not isExist:
        #     # Create a new directory because it does not exist
        #     os.makedirs(self._PARAMETERS['path_output_file'])

        # with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'w') as file:
        #     file.write(
        #         f"Network: {self._PARAMETERS['network_name']} \nnr of vs: {flownetwork.nr_of_vs} - nr of boundary vs: {len(flownetwork.boundary_vs)} "
        #         f"- nr of es: "
        #         f" {flownetwork.nr_of_es} \n")

        while flownetwork.convergence_check is False:

            iteration = flownetwork.iteration
            # ----- iterative routine -----
            if iteration > 0:
                self.iterative_routine(flownetwork)

            if iteration % 100 == 0 and iteration > 1:
                print(iteration)


            node_residual, node_relative_residual, local_balance_rbc, node_flow_change_total, indices_over_blue = flownetwork.node_residual, \
                flownetwork.node_relative_residual, \
                flownetwork.local_balance_rbc, flownetwork.node_flow_change_total, flownetwork.indices_over_blue

            match self._PARAMETERS['sor']:
                case 'sor':
                    if flownetwork.stop:
                        flownetwork.convergence_check = True

                        residual_plot(flownetwork, flownetwork.residualOverIterationMax, flownetwork.residualOverIterationNorm,
                                      self._PARAMETERS, " ", "", "convergence")
                        path = self._PARAMETERS['path_output_file'] + '/'
                        isExist = os.path.exists(path)
                        if not isExist:
                            os.makedirs(path)

                        f = open(path + '/' + self._PARAMETERS['network_name'] + '.pckl', 'wb')
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
                    elif iteration % 50 == 0 and iteration>0 :
                        residual_plot(flownetwork, flownetwork.residualOverIterationMax, flownetwork.residualOverIterationNorm,
                                      self._PARAMETERS, " ", "", "convergence")
                case 'Berg':
                    if iteration == 0:
                        flownetwork.bergIteration.append(None)
                        flownetwork.Berg1.append(None)
                        flownetwork.Berg2.append(None)
                        flownetwork.BergFirstPartEq.append(None)
                        flownetwork.BergPressure.append(None)
                        flownetwork.BergFlow.append(None)
                        flownetwork.BergHD.append(None)
                        flownetwork.BergSecondPartEq.append(None)
                    else:
                        residual_berg = berg_convergence(self, flownetwork)

                    if iteration > 0 and flownetwork.berg_criteria >= residual_berg:
                        flownetwork.convergence_check = True
                        residual_plot(flownetwork, flownetwork.residualOverIterationMax, flownetwork.residualOverIterationNorm,
                                      self._PARAMETERS, " ", "", "convergence")

                        residual_plot_berg(flownetwork, flownetwork.bergIteration, self._PARAMETERS, " ", "",
                                           "convergence_berg")

                        f = open(self._PARAMETERS['path_output_file'] + self._PARAMETERS['network_name'] + '.pckl', 'wb')
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

                case 'Rasmussen':
                    if (iteration > 0 and flownetwork.flow_convergence_criteria <= flownetwork.rasmussen_flow_threshold and
                            flownetwork.hd_convergence_criteria <= flownetwork.rasmussen_hd_threshold):
                        flownetwork.convergence_check = True

                        residual_plot(flownetwork, flownetwork.residualOverIterationMax, flownetwork.residualOverIterationNorm,
                                      self._PARAMETERS, " ",
                                      "",
                                      "convergence")
                        residual_plot_rasmussen(flownetwork, flownetwork.hd_convergence_criteria_plot, flownetwork.flow_convergence_criteria_plot,
                                                self._PARAMETERS, " ", "", "convergence_Rasmussen", flownetwork.rasmussen_hd_threshold,
                                                flownetwork.rasmussen_flow_threshold)

                        f = open(self._PARAMETERS['path_output_file'] + self._PARAMETERS['network_name'] + '.pckl', 'wb')
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

            flownetwork.iteration += 1

        print(f"Convergence: DONE in -> {flownetwork.iteration} \nAlpha -> {flownetwork.alpha} ")
