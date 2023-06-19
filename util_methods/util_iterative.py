# from testcases.testcase_blood_flow_iterative_model import flow_network
import sys

import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression

from util_methods.util_plot import s_curve_util, s_curve_personalized_thersholds, util_convergence_plot


def predictor_corrector_scheme(PARAMETERS, flownetwork, alpha, old):
    """

    """
    hematocrit = copy.copy(flownetwork.hd)
    new_hematocrit = np.zeros(len(hematocrit))
    for hemat in range(0, (len(hematocrit))):
        new_hematocrit[hemat] = (PARAMETERS["alpha"] * old[hemat]) + (
                (1 - PARAMETERS["alpha"]) * hematocrit[hemat])
    return new_hematocrit


def logifunc(x, a, b, c, d):
    return a / (1 + np.exp(-c * (x - d))) + b


def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))


def util_iterative_method(PARAMETERS, flownetwork):
    """
    Util to iterate with the method
    - it has been already performed a iteration with the common normal one (n=0) so
    now n=1
    in this case I skip the convergence and the corrector scheme

    I'll perform the corrector scheme and check at n=2

    """
    alpha = PARAMETERS["alpha"]

    flownetwork.convergence_check = False

    print("Convergence: ...")
    flownetwork.iteration = 0
    iteration_plot = []

    # first iteration

    while flownetwork.convergence_check is False:
        old_hematocrit = copy.copy(flownetwork.hd)
        old_flow = copy.copy(flownetwork.flow_rate)

        # iteration n=1
        flownetwork.iterative()

        # check if we are in convergences
        match PARAMETERS["convergence_case"]:
            case 1:
                convergence = np.abs(
                    (flownetwork.hd * np.abs(flownetwork.flow_rate)) - (old_hematocrit * np.abs(old_flow)))
                for element in convergence:
                    if element < PARAMETERS["epsilon"]:
                        flownetwork.convergence_check = True
                    else:
                        flownetwork.convergence_check = False
                        iteration_plot = np.append(iteration_plot, element)
                        flownetwork.iteration += 1
                        print("iteration " + str(flownetwork.iteration) + " " + str(iteration_plot))
                        break
            case 2:
                convergence = np.abs(flownetwork.flow_rate - old_flow) / old_flow
                if np.max(convergence) < PARAMETERS["epsilon_second_method"]:
                    flownetwork.convergence_check = True
                else:
                    flownetwork.convergence_check = False
                    iteration_plot = np.append(iteration_plot, np.max(convergence))
                    flownetwork.iteration += 1
                    print("iteration " + str(flownetwork.iteration) + " " + str(iteration_plot))

    # iteration_plot = np.append(iteration_plot, 0)
    iteration_plot = np.append(iteration_plot, 0)
    flownetwork.iteration += 1
    print("Error at each iteration " + str(iteration_plot))
    print("Convergence: DONE in -> " + str(flownetwork.iteration))

    util_convergence_plot(flownetwork, iteration_plot, PARAMETERS)

    s_curve_util(PARAMETERS, flownetwork)

    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.1)
    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.3)
    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.5)
    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.7)
