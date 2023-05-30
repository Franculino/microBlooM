# from testcases.testcase_blood_flow_iterative_model import flow_network
import sys

import numpy as np
import copy
import matplotlib.pyplot as plt


def predictor_corrector_scheme(flownetwork, alpha):
    """

    """
    hematocrit = flownetwork.hd
    new_hematocrit = np.zeros(len(hematocrit))

    for hemat in range(0, (len(hematocrit))):
        new_hematocrit[hemat] = (alpha * hematocrit[hemat]) + (
                (1 - alpha) * hematocrit[hemat])
    new_start = (alpha * flownetwork.ht_init) + (
            (1 - alpha) * flownetwork.ht_init)
    return new_hematocrit, new_start


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
    iteration = 1
    iteration_plot = []

    while flownetwork.convergence_check is False:
        old_hematocrit = copy.deepcopy(flownetwork.hd)
        old_flow = copy.deepcopy(flownetwork.flow_rate)
        old_init = copy.deepcopy(flownetwork.ht_init)
        # iteration n=1
        flownetwork.iterative()
        # now I have the new flow_rate and new_hematocrit
        # check if we are in convergences
        convergence = ((flownetwork.hd * flownetwork.flow_rate) - (old_hematocrit * old_flow))
        convergence_init = (old_init - flownetwork.ht_init)
        for element in convergence:
            if element < PARAMETERS["epsilon"]:
                flownetwork.convergence_check = True
            else:
                flownetwork.convergence_check = False
                iteration_plot = np.append(iteration_plot, element)
                break
        if convergence_init >= PARAMETERS["epsilon"]:
            flownetwork.convergence_check = False
        if iteration == 3:
            flownetwork.convergence_check = True

        if flownetwork.convergence_check is False:
            iteration += 1
            flownetwork.hd, flownetwork.ht_init = predictor_corrector_scheme(flownetwork, alpha)
            flownetwork.iterative()
    iteration_plot = np.append(iteration_plot, 0)
    print("Convergence: DONE in -> " + str(iteration))
    plt.style.use('seaborn-whitegrid')
    print(iteration_plot)
    # convergence plot
    plt.plot(range(0, iteration), iteration_plot,'-ok')
    plt.title("Convergence plot")
    plt.xlabel("Iteration")
    plt.ylabel("Error difference")
    plt.show()
