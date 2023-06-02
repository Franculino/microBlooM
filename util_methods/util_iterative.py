# from testcases.testcase_blood_flow_iterative_model import flow_network
import sys

import numpy as np
import copy
import matplotlib.pyplot as plt


def predictor_corrector_scheme(PARAMETERS, flownetwork, alpha):
    """

    """
    hematocrit = copy.deepcopy(flownetwork.hd)
    new_hematocrit = np.zeros(len(hematocrit))
    alpha = copy.deepcopy(PARAMETERS["alpha"])
    #    init = copy.deepcopy(flownetwork.ht_init)
    for hemat in range(0, (len(hematocrit))):
        new_hematocrit[hemat] = (alpha * hematocrit[hemat]) + (
                (1 - alpha) * hematocrit[hemat])

    return new_hematocrit


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
        # iteration n=1
        flownetwork.iterative()
        # now I have the new flow_rate and new_hematocrit
        # check if we are in convergences
        convergence = np.abs((flownetwork.hd * flownetwork.flow_rate) - (old_hematocrit * old_flow))

        for element in convergence:
            if element < PARAMETERS["epsilon"]:
                flownetwork.convergence_check = True
            else:
                flownetwork.convergence_check = False
                iteration_plot = np.append(iteration_plot, element)
                print(element)
                break
        # if convergence_init >= PARAMETERS["epsilon"]:
        #     flownetwork.convergence_check = False

        if flownetwork.convergence_check is False:
            iteration += 1
            flownetwork.hd = predictor_corrector_scheme(PARAMETERS, flownetwork, alpha)

    # to plot
    iteration_plot = np.append(iteration_plot, 0)
    print("Convergence: DONE in -> " + str(iteration))
    fig = plt.figure(figsize=(10, 8), dpi=80)
    ax = fig.add_subplot(111)
    plt.style.use('seaborn-whitegrid')
    print(iteration_plot)
    # convergence plot

    plt.title("Convergence plot")
    plt.xlabel("Iteration")
    plt.ylabel("Error difference")
    plt.plot(range(0, iteration), iteration_plot, '-ok')
    for i, v in enumerate(iteration_plot):
        ax.annotate(str("{:.2e}".format(v)), xy=(i, v), xytext=(-7, 7), textcoords='offset points')
    plt.show()
