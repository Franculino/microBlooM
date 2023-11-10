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

            # ----- iterative routine -----

            if flownetwork.iteration % 150 == 0 and flownetwork.iteration > 2:
                residual_plot(flownetwork, flownetwork.residualOverIterationMax, flownetwork.residualOverIterationNorm, flownetwork._PARAMETERS, " ", "",
                              "convergence")

            if flownetwork.maxBalance <= flownetwork.two_MagnitudeThreshold and flownetwork.boundary_hematocrit[0] != flownetwork.goal:
                #
                # TODO: if we want to force
                # it 1 and
                # put
                flownetwork.boundary_hematocrit = np.full(len(flownetwork.boundary_vs), (flownetwork.boundary_hematocrit[0] + 0.01))
                flownetwork.increment += 1
                flownetwork.alpha = 0.5
                # --- save variables ---
                f = open(str(self._PARAMETERS['network_name']) +str(flownetwork.iteration) + '.pckl', 'wb')
                pickle.dump(
                    [flownetwork.flow_rate, flownetwork.node_relative_residual, flownetwork.positions_of_elements_not_in_boundary, flownetwork.node_residual,
                     flownetwork.two_MagnitudeThreshold, flownetwork.node_flow_change, flownetwork.vessel_flow_change,
                     flownetwork.node_relative_residual_plot, flownetwork.indices_over_blue, flownetwork.node_flow_change_total], f)
                f.close()

            elif flownetwork.boundary_hematocrit[0] == flownetwork.goal:
                # --- save variables ---
                f = open(str(self._PARAMETERS['network_name']) + '.pckl', 'wb')
                pickle.dump(
                    [flownetwork.flow_rate, flownetwork.node_relative_residual, flownetwork.positions_of_elements_not_in_boundary, flownetwork.node_residual,
                     flownetwork.two_MagnitudeThreshold, flownetwork.node_flow_change, flownetwork.vessel_flow_change,
                     flownetwork.node_relative_residual_plot, flownetwork.indices_over_blue, flownetwork.node_flow_change_total], f)
                f.close()

                flownetwork.convergence_check = True
                residual_plot(flownetwork, flownetwork.residualOverIterationMax, flownetwork.residualOverIterationNorm, flownetwork._PARAMETERS,
                              str(self._PARAMETERS['network_name']), "", "final_convergence_plot")

                # --- ALL NODES ---
                frequency_plot(flownetwork, flownetwork.node_relative_residual_plot, 'Relative Residual', 'relative residual', 'seagreen', 10000,
                               "all_node")
                frequency_plot(flownetwork, flownetwork.node_residual_plot, 'Residual', 'residual', 'skyblue', 100000, "all_node")

                # --- NON CONVERGING NODES ---
                frequency_plot(flownetwork, flownetwork.node_relative_residual_plot[flownetwork.indices_over_blue], 'Relative Residual',
                               'relative residual', 'seagreen', 10000, "non_converging")
                frequency_plot(flownetwork, flownetwork.node_residual_plot[flownetwork.indices_over_blue], 'Residual', 'residual',
                               'skyblue', 100000, "non_converging")

                # --- NODE WITH FLOW CHANGE BEHAVIOUR ---
                frequency_plot(flownetwork, flownetwork.node_relative_residual_plot[flownetwork.node_flow_change_total], 'Relative Residual',
                               'relative residual', 'seagreen', 10000, "flow_change_total")
                frequency_plot(flownetwork, flownetwork.node_residual_plot[flownetwork.node_flow_change_total], 'Residual', 'residual',
                               'skyblue', 100000, "flow_change_total")

                # --- NON-CONVERGING WITHOUT FLOW DIRECTION CHANGE ---
                mask = np.ones_like(flownetwork.node_relative_residual_plot, dtype=bool)
                mask[flownetwork.node_flow_change_total] = False
                result_array = flownetwork.node_relative_residual_plot[mask]

                frequency_plot(flownetwork, result_array, 'Relative Residual', 'relative residual', 'seagreen', 10000, "non_converging_without_flow")

                mask = np.ones_like(flownetwork.node_residual_plot, dtype=bool)
                mask[flownetwork.node_flow_change_total] = False
                result_array2 = flownetwork.node_relative_residual_plot[mask]
                frequency_plot(flownetwork, result_array2, 'Residual', 'residual', 'skyblue', 100000, "non_converging_without_flow")


            else:
                flownetwork.convergence_check = False

        print(f"Convergence: DONE in -> {flownetwork.iteration} \nAlpha -> {flownetwork.alpha} ")
