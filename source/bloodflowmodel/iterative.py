import pickle
from abc import ABC, abstractmethod
import os
import warnings
from types import MappingProxyType

import numpy as np


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
            [flownetwork.flow_rate,
             flownetwork.local_balance_rbc,
             flownetwork.two_MagnitudeThreshold,
             flownetwork.pressure,
             flownetwork.hd,
             flownetwork.iteration,
             flownetwork.alpha], f)
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
        pass


class IterativeRoutineMultipleIteration(IterativeRoutine):
    """
    Iterative approach based on our convergence criteria, Berg and Rasmussen
    - our convergence criteria:
        - value: max value of RBCs-flow balance
        - threshold: defined as two magnitude higher that the Zero Flow threshold for the respective network
    - Berg convergence criteria:
        - value: custom residual defined in berg_convergence()
        - threshold: 1e-6
    - Rasmussen convergence criteria:
        - value: Flow and Hematocrit absolute difference
        - threshold: defined as eight magnitude lower than the initial value of both values
    """

    def _iterative_method(self, flownetwork):
        """
        Iterative method used to iterate over.
        The different criteria are selected via a match case.

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

                match self._PARAMETERS['iterative_routine']:

                    # OUR
                    case 2:
                        # Convergence criteria analyzed in flow_balance
                        if flownetwork.our_convergence_criteria:
                            # to exit from the loop if the criteria is fulfilled
                            flownetwork.convergence_check = True
                            # TODO: residual over the iteration they alha and residual
                            # iteration and reisual value
                            # cvs file with it and residual value
                        else:
                            flownetwork.iteration += 1

                    # BERG
                    case 3:
                        # Residual of Berg
                        residual_berg = self.berg_convergence(flownetwork)
                        # Convergence criteria
                        if flownetwork.berg_criteria >= residual_berg:
                            # to exit from the loop if the criteria is fulfilled
                            flownetwork.convergence_check = True
                        else:
                            flownetwork.iteration += 1

                    # Rasmussen
                    case 4:
                        # Convergence Criteria
                        if flownetwork.flow_convergence_criteria <= flownetwork.rasmussen_flow_threshold and flownetwork.hd_convergence_criteria <= flownetwork.rasmussen_hd_threshold:
                            # to exit from the loop if the criteria is fulfilled
                            flownetwork.convergence_check = True
                        else:
                            flownetwork.iteration += 1
                # Save pckl if selected at the last iteration
                if flownetwork.convergence_check:
                    # CSV to save iteration, residual value and alpha value
                    match self._PARAMETERS['iterative_routine']:
                        # OUR
                        case 2:
                            scores = ['Our', flownetwork.maxBalance, flownetwork.two_MagnitudeThreshold, flownetwork.iteration, flownetwork.alpha]
                        # Berg
                        case 3:
                            scores = ['Berg', residual_berg, flownetwork.berg_criteria, flownetwork.iteration, flownetwork.alpha]
                        # Rasmussen
                        case 4:
                            scores = ['Rasmussen [flow, HD]', [flownetwork.flow_convergence_criteria, flownetwork.hd_convergence_criteria],
                                      [flownetwork.rasmussen_flow_threshold, flownetwork.rasmussen_hd_threshold], flownetwork.iteration, flownetwork.alpha]

                    names = ['Method', 'Residual Value', 'Threshold', 'Iteration', 'Alpha']
                    np.savetxt(self._PARAMETERS['path_output_file'] + 'RESULT.csv', [names, scores], delimiter=',', fmt='%s')

                    if self._PARAMETERS['pckl_save']:
                        self.save_pckl_data(flownetwork, path_pckl=self._PARAMETERS['path_output_file'] + self._PARAMETERS['network_name'] + '/')

            else:
                match self._PARAMETERS['iterative_routine']:
                    # OUR
                    case 2:
                        # Set at the first iteration the starting value of alpha
                        flownetwork.alpha = 1
                    # Berg
                    case 3:
                        # Set at the first iteration the starting value of alpha
                        flownetwork.alpha = 0.2
                    # Rasmussen
                    case 4:
                        # Set at the first iteration the starting value of alpha
                        flownetwork.alpha = 1
                # Flow Balance: in the iterative is automatically check, but at the first iteration (0) it is necessary
                flownetwork.check_flow_balance()
                flownetwork.iteration += 1

        print(f"Convergence: DONE in -> {flownetwork.iteration}")

    def berg_convergence(self, flownetwork):
        """
        Method to calculate the berg convergence
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # First initialization
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
