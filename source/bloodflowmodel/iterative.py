import pickle
from abc import ABC, abstractmethod
import os
import warnings
from types import MappingProxyType


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

    @staticmethod
    def iterative_routine(flownetwork):
        """
        Call the functions that solve for the pressures and flow rates.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :ack: burn baby burn (A.S.)
        """
        flownetwork.update_transmissibility()
        flownetwork.imp_buildsystem.build_linear_system(flownetwork)
        flownetwork.imp_solver.update_pressure_flow(flownetwork)
        flownetwork.imp_rbcvelocity.update_velocity(flownetwork)
        flownetwork.check_flow_balance()

    @staticmethod
    def save_pckl_data(flownetwork, path_pckl):
        """
        Function to save the main value necessary in a .pckl file

        @param flownetwork: flow network object
        @type flownetwork: source.flow_network.FlowNetwork
        @param path_pckl: path to where to store the variables
        @type path_pckl: string

        """

        # Check if the file already exists and creates it
        isExist = os.path.exists(path_pckl)
        if not isExist:
            os.makedirs(path_pckl)

        # Open the stream and store the pckl
        f = open(path_pckl + 'store_variable' + '.pckl', 'wb')
        pickle.dump(
            [abs(flownetwork.flow_rate),
             flownetwork.node_relative_residual,
             flownetwork.positions_of_elements_not_in_boundary,
             flownetwork.local_balance_rbc,
             flownetwork.two_MagnitudeThreshold,
             flownetwork.node_flow_change,
             flownetwork.vessel_flow_change,
             flownetwork.indices_over_blue,
             flownetwork.node_flow_change_total,
             flownetwork.vessel_flow_change_total,
             flownetwork.pressure,
             flownetwork.hd], f)
        f.close()


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


class IterativeRoutineMultipleIteration(IterativeRoutine):
    """
    Iterative approach based on our definition
    """

    def _iterative_method(self, flownetwork):
        """
        Iterative method based on the convergence criteria of Max Residual under a certain threshold

        @param flownetwork: flow network object
        @type flownetwork: source.flow_network.FlowNetwork
        """

        # warning handled for np.nan and np.inf
        warnings.filterwarnings("ignore")
        flownetwork.convergence_check = False

        print("Convergence: ...")

        while flownetwork.convergence_check is False:

            iteration = flownetwork.iteration
            # ----- iterative routine -----
            if iteration > 0:
                self.iterative_routine(flownetwork)
                # Convergence criteria analyzed in flow_balance
                if flownetwork.our_convergence_criteria:
                    # to exit from the loop if the criteria is fulfilled
                    flownetwork.convergence_check = True
                    # Save pckl if selected
                    if self._PARAMETERS['pckl_save']:
                        self.save_pckl_data(flownetwork, path_pckl=self._PARAMETERS['path_output_file'] + self._PARAMETERS['network_name'] + '/')

            else:
                # Set at the first iteration the starting value of alpha
                flownetwork.alpha = 1
                # Flow Balance
                flownetwork.check_flow_balance()

            flownetwork.iteration += 1

        print(f"Convergence: DONE in -> {flownetwork.iteration}")


class IterativeRoutineBerg(IterativeRoutine):
    """
    Iterative method based on the convergence criteria define by the PhD's thesis of Berg
    """

    def _iterative_method(self, flownetwork):
        """
        Iterative method based on the convergence criteria of the desiged "residual" from Berg

        @param flownetwork: flow network object
        @type flownetwork: source.flow_network.FlowNetwork
        """

        # warning handled for np.nan and np.inf
        warnings.filterwarnings("ignore")
        flownetwork.convergence_check = False

        print("Convergence: ...")

        while flownetwork.convergence_check is False:
            # Variable needed for the computation
            iteration = flownetwork.iteration

            # ----- iterative routine -----
            if iteration > 0:
                self.iterative_routine(flownetwork)
                # ----- iterative routine -----

                # Residual of Berg
                residual_berg = self.berg_convergence(flownetwork)
                # Convergence criteria
                if flownetwork.berg_criteria >= residual_berg:
                    # to exit from the loop if the criteria is fulfilled
                    flownetwork.convergence_check = True
                    # Save pckl if selected
                    if self._PARAMETERS['pckl_save']:
                        self.save_pckl_data(flownetwork, path_pckl=self._PARAMETERS['path_output_file'] + self._PARAMETERS['network_name'] + '/')

            else:
                # Set at the first iteration the starting value of alpha
                flownetwork.alpha = 0.2
                # Flow Balance
                flownetwork.check_flow_balance()

            flownetwork.iteration += 1

        print(f"Convergence: DONE in -> {flownetwork.iteration}")

    def berg_convergence(self, flownetwork):
        """
        Method to calculate the berg convergence
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # First initialation
        if flownetwork.iteration == 0:
            flownetwork.bergIteration.append(None)
            flownetwork.Berg1.append(None)
            flownetwork.Berg2.append(None)
            flownetwork.BergFirstPartEq.append(None)
            flownetwork.BergPressure.append(None)
            flownetwork.BergFlow.append(None)
            flownetwork.BergHD.append(None)
            flownetwork.BergSecondPartEq.append(None)
            residual = None
        else:
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


class IterativeRoutineRasmussen(IterativeRoutine):
    """
    Iterative method based on the convergence criteria define by the Rasmussen 2018
    """

    def _iterative_method(self, flownetwork):
        """
        Iterative method based on the absolute flow and hematocrit change under a certain threshold

        @param flownetwork: flow network object
        @type flownetwork: source.flow_network.FlowNetwork
        """

        # warning handled for np.nan and np.inf
        warnings.filterwarnings("ignore")
        flownetwork.convergence_check = False

        print("Convergence: ...")

        while flownetwork.convergence_check is False:
            # Variable needed for the computation
            iteration = flownetwork.iteration

            # ----- iterative routine -----
            if iteration > 0:
                self.iterative_routine(flownetwork)

                # Convergence Criteria
                if flownetwork.flow_convergence_criteria <= flownetwork.rasmussen_flow_threshold and flownetwork.hd_convergence_criteria <= flownetwork.rasmussen_hd_threshold:
                    # to exit from the loop if the criteria is fulfilled
                    flownetwork.convergence_check = True
                    # Save pckl if selected
                    if self._PARAMETERS['pckl_save']:
                        self.save_pckl_data(flownetwork, path_pckl=self._PARAMETERS['path_output_file'] + self._PARAMETERS['network_name'] + '/')

            else:
                # Set at the first iteration the starting value of alpha
                flownetwork.alpha = 1
                # Flow Balance
                flownetwork.check_flow_balance()

            flownetwork.iteration += 1

        print(f"Convergence: DONE in -> {flownetwork.iteration}")
