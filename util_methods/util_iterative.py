# from testcases.testcase_blood_flow_iterative_model import flow_network
import sys

import numpy as np
import copy
import matplotlib.pyplot as plt
from line_profiler_pycharm import profile
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression

from util_methods.util_plot import s_curve_util, s_curve_personalized_thersholds, util_convergence_plot, s_curve_util_trifurcation, util_display_graph, graph_creation


def predictor_corrector_scheme(PARAMETERS, flownetwork, old_hemat):
    hematocrit = copy.deepcopy(flownetwork.hd)
    new_hematocrit = np.zeros(len(hematocrit))
    for hemat in range(0, (len(hematocrit))):
        new_hematocrit[hemat] = (PARAMETERS["alpha"] * old_hemat[hemat]) + (
                (1 - PARAMETERS["alpha"]) * hematocrit[hemat])
    return new_hematocrit


@profile
def util_iterative_method(PARAMETERS, flownetwork, flow_balance):
    """
    Util to iterate with the method
    - it has been already performed iteration with the common normal one (n=0) so
    now n=1
    in this case I skip the convergence and the corrector scheme
    I'll perform the corrector scheme and check at n=2

    """
    flownetwork.convergence_check = False

    print("Convergence: ...")
    flownetwork.iteration = 0
    iteration_plot = []

    # first iteration

    while flownetwork.convergence_check is False:
        old_hematocrit = copy.copy(flownetwork.hd)
        old_flow = np.abs(copy.copy(flownetwork.flow_rate))

        # iteration n=1
        flownetwork.iterative_part_one()
        # flownetwork.hd = predictor_corrector_scheme(PARAMETERS, flownetwork, old_hematocrit)
        flownetwork.iterative_part_two()

        # check if we are in convergences
        match PARAMETERS["convergence_case"]:
            case 1:
                # convergence = np.abs((flownetwork.hd * np.abs(flownetwork.flow_rate)) - (old_hematocrit * np.abs(old_flow)))
                convergence = np.average(flownetwork.hd * np.abs(flownetwork.flow_rate) - (old_hematocrit * np.abs(old_flow)))
                if convergence < PARAMETERS["epsilon"]:
                    flownetwork.convergence_check = True
                else:
                    flownetwork.convergence_check = False
                    iteration_plot = np.append(iteration_plot, convergence)
                    flownetwork.iteration += 1

                    if flownetwork.iteration % 50 == 0:
                        print("iteration " + str(flownetwork.iteration) + " " + str(convergence))
                        util_convergence_plot(flownetwork, iteration_plot, PARAMETERS)
                # for element in convergence:
                #     if element < PARAMETERS["epsilon"]:
                #         flownetwork.convergence_check = True
                #     else:
                #         flownetwork.convergence_check = False
                #         iteration_plot = np.append(iteration_plot, element)
                #         flownetwork.iteration += 1
                #         print("iteration " + str(flownetwork.iteration) + " " + str(iteration_plot))
                #         break
            case 2:
                convergence = np.abs(np.average(np.abs(flownetwork.flow_rate)) - np.average(old_flow)) / np.average(old_flow)

                if convergence < PARAMETERS["epsilon_second_method"]:
                    flownetwork.convergence_check = True
                else:
                    flownetwork.convergence_check = False
                    iteration_plot = np.append(iteration_plot, convergence)
                    flownetwork.iteration += 1

                    if flownetwork.iteration % 50 == 0:
                        print("iteration " + str(flownetwork.iteration) + " " + str(convergence))
                        util_convergence_plot(flownetwork, iteration_plot, PARAMETERS)

    # iteration_plot = np.append(iteration_plot, 0)
    iteration_plot = np.append(iteration_plot, 0)
    flownetwork.iteration += 1
    print("Error at each iteration " + str(iteration_plot))
    print("Convergence: DONE in -> " + str(flownetwork.iteration))

    # util_display_graph(graph_creation(flownetwork), PARAMETERS, flownetwork)

    util_convergence_plot(flownetwork, iteration_plot, PARAMETERS)

    s_curve_util(PARAMETERS, flownetwork)

    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.1)
    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.3)
    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.5)
    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.7)

    s_curve_util_trifurcation(PARAMETERS, flownetwork)
