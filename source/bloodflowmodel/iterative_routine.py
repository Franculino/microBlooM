# from testcases.testcase_blood_flow_iterative_model import flow_network
import sys

import numpy as np
import copy
from line_profiler_pycharm import profile

from source.fileio.create_display_plot import s_curve_util, s_curve_personalized_thersholds, util_convergence_plot, s_curve_util_trifurcation, frequency_plot


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
    convergence_percentual = 1
    iteration_plot = []
    itr_min_c = 0
    itr_min_data = 0
    # first iteration

    while flownetwork.convergence_check is False:
        # Old hematocrit and flow to be used after in the convergence
        old_hematocrit = copy.deepcopy(flownetwork.hd)
        old_flow = np.abs(copy.deepcopy(flownetwork.flow_rate))

        # flownetwork.hd = predictor_corrector_scheme(PARAMETERS, flownetwork, old_hematocrit)
        flownetwork.iterative_approach()
        flow_balance.check_flow_balance()
        flow_rate = np.abs(flownetwork.flow_rate)

        # CONVERGENCE
        # normalize by the maximum value
        # different order of magnitude
        # many ituliers
        cnvg_flow = np.abs(flow_rate - old_flow)
        conv_avg_flow = np.average(cnvg_flow) / np.max(old_flow) * 100
        conv_max_flow = np.max(cnvg_flow) / np.max(old_flow) * 100

        cnvg_rbc = np.abs((flownetwork.hd * flow_rate) - (old_hematocrit * old_flow))
        conv_avg_rbc = np.mean(cnvg_rbc) / np.max(old_hematocrit * old_flow) * 100
        conv_max_rbc = np.max(cnvg_rbc) / np.max(old_hematocrit * old_flow) * 100
        plus = np.sum(cnvg_rbc)/len(cnvg_rbc)
        flownetwork.iteration += 1

        if conv_avg_flow < 0.5 and conv_avg_rbc < 0.5 and conv_max_flow < 1 and conv_max_rbc < 1:
            flownetwork.convergence_check = True
            print(f"Iteration number {flownetwork.iteration} and data: {conv_avg_flow, conv_max_flow, conv_avg_rbc, conv_max_rbc}")
        else:
            flownetwork.convergence_check = False
            print(f"Iteration number {flownetwork.iteration} and data: {conv_avg_flow, conv_max_flow, conv_avg_rbc, conv_max_rbc}")

    print("Convergence: DONE in -> " + str(flownetwork.iteration))


    # s_curve_util(PARAMETERS, flownetwork)
    #
    # s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.1)
    # s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.3)
    # s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.5)
    # s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.7)
    #
    # s_curve_util_trifurcation(PARAMETERS, flownetwork)
