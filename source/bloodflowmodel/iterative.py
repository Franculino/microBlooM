import sys
from abc import ABC, abstractmethod
import os
import warnings
import numpy as np
import copy
from types import MappingProxyType

from source.fileio.create_display_plot import s_curve_util, s_curve_personalized_thersholds, util_convergence_plot, s_curve_util_trifurcation, util_convergence_plot_final, \
    percentage_vessel_plot


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
        flownetwork.iteration, flownetwork.cnvg_rbc, flownetwork.cnvg_flow = 0, 0, 0
        # to reconstruct the position of the value
        # position_array = np.array([i for i in range(flownetwork.nr_of_es)])
        save_data_max_flow, save_data_max_rbc, save_data_max_hemat = [], [], []
        save_data_avg_flow, save_data_avg_rbc, save_data_avg_hemat = [], [], []
        save_data_vessel_flow, save_data_vessel_rbc, save_data_vessel_hemat = [], [], []

        isExist = os.path.exists(self._PARAMETERS['path_output_file'])
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self._PARAMETERS['path_output_file'])

        with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'w') as file:
            file.write(f"Network: {self._PARAMETERS['network_name']} \nnr of vs: {flownetwork.nr_of_vs} - nr of boundary vs: {len(flownetwork.boundary_vs)} - nr of es:"
                       f" {flownetwork.nr_of_es} \n")

        while flownetwork.convergence_check is False:

            # Old hematocrit and flow to be used after in the convergence
            old_hematocrit = copy.deepcopy(flownetwork.hd)
            old_flow = np.abs(copy.deepcopy(flownetwork.flow_rate))

            # ----- iterative routine -----
            self.iterative_routine(flownetwork)

            # ----- iterative routine -----

            # start converging stuff
            flow_rate = np.abs(copy.deepcopy(flownetwork.flow_rate))
            hd = copy.deepcopy(flownetwork.hd)

            # flow data
            cnvg_flow = np.abs(old_flow - flow_rate) / old_flow * 100

            # per avere sotto 5
            # cnvg_flow = np.where(cnvg_flow >= 5, 0, cnvg_flow)

            # mask to filter out nan/inf
            flow_mask = cnvg_flow[np.isfinite(cnvg_flow)]

            # to reconstruct the position of the value

            cnvg_flow_avg_per = np.average(flow_mask)
            key_cnvg_flow_max = np.argmax(flow_mask)
            cnvg_flow_max_per = flow_mask[key_cnvg_flow_max]

            # RBCs
            cnvg_rbc = np.abs((hd * flow_rate) - (old_hematocrit * old_flow)) / np.abs(old_hematocrit * old_flow) * 100
            cnvg_rbc_avg_per = np.average(cnvg_rbc[np.isfinite(cnvg_rbc)])
            cnvg_rbc_max_per = np.max(cnvg_rbc[np.isfinite(cnvg_rbc)])

            # RBCs
            cnvg_hem = np.abs(old_hematocrit - flownetwork.hd) / old_hematocrit * 100

            cnvg_hem_avg_per = np.average(cnvg_hem[np.isfinite(cnvg_hem)])
            cnvg_hem_max_per = np.max(cnvg_hem[np.isfinite(cnvg_hem)])

            save_data_max_flow, save_data_max_rbc, save_data_max_hemat = np.append(save_data_max_flow, cnvg_flow_max_per), \
                np.append(save_data_max_rbc, cnvg_rbc_max_per), np.append(save_data_max_hemat, cnvg_hem_max_per)

            save_data_avg_flow, save_data_avg_rbc, save_data_avg_hemat = np.append(save_data_avg_flow, cnvg_flow_avg_per), \
                np.append(save_data_avg_rbc, cnvg_rbc_avg_per), np.append(save_data_avg_hemat, cnvg_hem_avg_per)

            vessel_flow, vessel_rbc, vessel_hd = len(cnvg_flow[cnvg_flow >= 1]) / flownetwork.nr_of_es * 100, len(cnvg_rbc[cnvg_rbc >= 1]) / flownetwork.nr_of_es * 100, \
                                                 len(cnvg_hem[cnvg_hem >= 1]) / flownetwork.nr_of_es * 100

            save_data_vessel_flow, save_data_vessel_rbc, save_data_vessel_hemat = np.append(save_data_vessel_flow, vessel_flow), \
                np.append(save_data_vessel_rbc, vessel_rbc), np.append(save_data_vessel_hemat, vessel_hd)

            flownetwork.iteration += 1

            if cnvg_hem_avg_per < 0.5 and cnvg_hem_max_per < 1:  # (cnvg_flow_avg_per < 0.5 and cnvg_flow_max_per < 1 and cnvg_rbc_avg_per < 0.5 and cnvg_rbc_max_per < 1 and
                with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'a') as file:
                    file.write(f"\n-----------------------------\nIteration number {flownetwork.iteration} and data: "
                               f"FLOW {cnvg_flow_avg_per:.5e} {cnvg_flow_max_per:.5e}  RBC {cnvg_rbc_avg_per:.5e} {cnvg_rbc_max_per:.5e}  HEMATOCRIT {cnvg_hem_avg_per:.5e}"
                               f" {cnvg_hem_max_per:.5e} ALPHA {flownetwork.alpha} VESSEL >1% flow:{vessel_flow:.5e}% rbc:{vessel_rbc:.5e}% hd:{vessel_hd:.5e}%\n-----------------------------")

                util_convergence_plot_final(flownetwork, save_data_max_flow, self._PARAMETERS, f"Convergence plot of max flow error(%) after {flownetwork.iteration}",
                                            "final/", "final_flow_max")
                util_convergence_plot_final(flownetwork, save_data_max_rbc, self._PARAMETERS, f"Convergence plot of max rbc error(%) after {flownetwork.iteration}",
                                            "final/", "final_rbc_max")
                util_convergence_plot_final(flownetwork, save_data_max_hemat, self._PARAMETERS, f"Convergence plot of max hemat error(%) after {flownetwork.iteration}",
                                            "final/", "final_hemat_max")

                util_convergence_plot_final(flownetwork, save_data_avg_flow, self._PARAMETERS, f"Convergence plot of average flow error(%) after {flownetwork.iteration}",
                                            "final/", "final_flow_avg")
                util_convergence_plot_final(flownetwork, save_data_avg_rbc, self._PARAMETERS, f"Convergence plot of average rbc error(%) after {flownetwork.iteration}",
                                            "final/", "final_rbc_avg")
                util_convergence_plot_final(flownetwork, save_data_avg_hemat, self._PARAMETERS, f"Convergence plot of average hemat error(%) after {flownetwork.iteration}",
                                            "final/", "final_hemat_avg")

                # util_convergence_plot(flownetwork, save_data_max_flow[-100:], self._PARAMETERS, f" flow error % max for all vessel", "final/", "flow_max")
                # util_convergence_plot(flownetwork, save_data_max_rbc[-100:], self._PARAMETERS, f" rbc error % max for all vessel", "final/", "rbc_max")
                # util_convergence_plot(flownetwork, save_data_max_hemat[-100:], self._PARAMETERS, f" hemat error % max for all vessel", "final/", "hemat_max")
                # util_convergence_plot(flownetwork, save_data_avg_flow[-100:], self._PARAMETERS, f" flow error % avg for all vessel", "final/", "flow_avg")
                # util_convergence_plot(flownetwork, save_data_avg_rbc[-100:], self._PARAMETERS, f" rbc error % avg for all vessel", "final/", "rbc_avg")
                # util_convergence_plot(flownetwork, save_data_avg_hemat[-100:], self._PARAMETERS, f" hemat error % avg for all vessel", "final/", "hemat_avg")
                #
                flownetwork.convergence_check = True

                percentage_vessel_plot(flownetwork, save_data_vessel_flow, self._PARAMETERS, f" Percentage of vessel with change under the threshold - FLOW",
                                       "vessel_flow")
                percentage_vessel_plot(flownetwork, save_data_vessel_hemat, self._PARAMETERS, f" Percentage of vessel with change under the threshold - HEMATOCRIT",
                                       "vessel_hd")
                percentage_vessel_plot(flownetwork, save_data_vessel_rbc, self._PARAMETERS, f" Percentage of vessel with change under the threshold - RBC", "vessel_rbc")


            else:

                if flownetwork.iteration % 100 == 0:
                    print(f"Iteration {flownetwork.iteration}")
                    util_convergence_plot(flownetwork, save_data_max_flow[-100:], self._PARAMETERS, f" flow error % max for all vessel", "max/", "flow_max")
                    util_convergence_plot(flownetwork, save_data_max_rbc[-100:], self._PARAMETERS, f" rbc error % max for all vessel", "max/", "rbc_max")
                    util_convergence_plot(flownetwork, save_data_max_hemat[-100:], self._PARAMETERS, f" hemat error % max for all vessel", "max/", "hemat_max")
                    util_convergence_plot(flownetwork, save_data_avg_flow[-100:], self._PARAMETERS, f" flow error % avg for all vessel", "average/", "flow_avg")
                    util_convergence_plot(flownetwork, save_data_avg_rbc[-100:], self._PARAMETERS, f" rbc error % avg for all vessel", "average/", "rbc_avg")
                    util_convergence_plot(flownetwork, save_data_avg_hemat[-100:], self._PARAMETERS, f" hemat error % avg for all vessel", "average/", "hemat_avg")

                    percentage_vessel_plot(flownetwork, save_data_vessel_flow, self._PARAMETERS, f" Percentage of vessel with change under the threshold - FLOW",
                                           "vessel_flow")
                    percentage_vessel_plot(flownetwork, save_data_vessel_hemat, self._PARAMETERS, f" Percentage of vessel with change under the threshold - HEMATOCRIT",
                                           "vessel_hd")
                    percentage_vessel_plot(flownetwork, save_data_vessel_rbc, self._PARAMETERS, f" Percentage of vessel with change under the threshold - RBC", "vessel_rbc")

                if flownetwork.iteration % 5 == 0:
                    with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'a') as file:
                        file.write(f"\nItr {flownetwork.iteration} and data: "
                                   f"FLOW {cnvg_flow_avg_per:.5e} {cnvg_flow_max_per:.5e}  RBC {cnvg_rbc_avg_per:.5e} {cnvg_rbc_max_per:.5e}  HEMATOCRIT {cnvg_hem_avg_per:.5e}"
                                   f" {cnvg_hem_max_per:.5e} ALPHA {flownetwork.alpha} VESSEL >1% flow:{vessel_flow:.5e}% rbc:{vessel_rbc:.5e}% hd:{vessel_hd:.5e}% ")

                flownetwork.convergence_check = False

        print(f"Convergence: DONE in -> {flownetwork.iteration} \nAlpha -> {flownetwork.alpha} ")

        s_curve_util(self._PARAMETERS, flownetwork)

        s_curve_personalized_thersholds(flownetwork, self._PARAMETERS, 0.1)
        s_curve_personalized_thersholds(flownetwork, self._PARAMETERS, 0.3)
        s_curve_personalized_thersholds(flownetwork, self._PARAMETERS, 0.5)
        s_curve_personalized_thersholds(flownetwork, self._PARAMETERS, 0.7)
        s_curve_personalized_thersholds(flownetwork, self._PARAMETERS, 0.9)

        s_curve_util_trifurcation(self._PARAMETERS, flownetwork)
