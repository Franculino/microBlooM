# from testcases.testcase_blood_flow_iterative_model import flow_network
import sys
import warnings
import numpy as np
import copy

from source.fileio.create_display_plot import s_curve_util, s_curve_personalized_thersholds, util_convergence_plot, s_curve_util_trifurcation, frequency_plot


def predictor_corrector_scheme(PARAMETERS, flownetwork, old_hemat):
    hematocrit = copy.deepcopy(flownetwork.hd)
    new_hematocrit = np.zeros(len(hematocrit))
    for hemat in range(0, (len(hematocrit))):
        new_hematocrit[hemat] = (PARAMETERS["alpha"] * old_hemat[hemat]) + (
                (1 - PARAMETERS["alpha"]) * hematocrit[hemat])
    return new_hematocrit


def util_iterative_method(PARAMETERS, flownetwork, flow_balance):
    """
    Util to iterate with the method
    - it has been already performed iteration with the common normal one (n=0) so
    now n=1
    in this case I skip the convergence and the corrector scheme
    I'll perform the corrector scheme and check at n=2

    """
    # warning handled for np.nan and np.inf
    warnings.filterwarnings("ignore")

    flownetwork.convergence_check = False

    print("Convergence: ...")
    flownetwork.iteration, flownetwork.cnvg_rbc, flownetwork.cnvg_flow = 0, 0, 0
    # to reconstruct the position of the value
    position_array = np.array([i for i in range(flownetwork.nr_of_es)])
    save_data = []
    # save_data = np.append(save_data, flownetwork.flow_rate[1876])

    with open(PARAMETERS['path_output_file'], 'a') as file:
        file.write(f"Network: {PARAMETERS['network_name']} \n- nr of vs: {flownetwork.nr_of_vs} - nr of boundary vs: {len(flownetwork.boundary_vs)} - nr of es:"
                   f" {flownetwork.nr_of_es} \n")

    while flownetwork.convergence_check is False:

        # Old hematocrit and flow to be used after in the convergence
        old_hematocrit = copy.deepcopy(flownetwork.hd)
        old_flow = np.abs(copy.deepcopy(flownetwork.flow_rate))
        # iterative routine
        flownetwork.iterative_approach()
        flow_balance.check_flow_balance()

        # start converging stuff
        flow_rate = np.abs(copy.deepcopy(flownetwork.flow_rate))
        hd = copy.deepcopy(flownetwork.hd)

        # flow data
        cnvg_flow_per = (np.abs(old_flow - flow_rate) / old_flow) * 100

        # per avere sotto 5
        # cnvg_flow_per = np.where(cnvg_flow_per >= 5, 0, cnvg_flow_per)

        # mask to filter out nan/inf
        flow_mask = cnvg_flow_per[np.isfinite(cnvg_flow_per)]

        # to reconstruct the position of the value
        position_array_mask = np.arange(len(cnvg_flow_per))[np.isfinite(cnvg_flow_per)]  # position_array[np.isfinite(cnvg_flow_per)]
        # print(f"diff between old and new position array: {len(position_array) - len(position_array_mask)} element")
        cnvg_flow_avg_per = np.average(flow_mask)  # np.abs(np.average(cnvg_flow_old) - np.average(cnvg_flow)) / np.average(cnvg_flow_old) * 100
        key_cnvg_flow_max = np.argmax(flow_mask)
        cnvg_flow_max_per = flow_mask[key_cnvg_flow_max]  # np.abs(np.max(cnvg_flow_old) - np.max(cnvg_flow)) / np.max(cnvg_flow_old) * 100

        # RBCs
        cnvg_rbc = np.abs((hd * flow_rate) - (old_hematocrit * old_flow)) / np.abs(old_hematocrit * old_flow) * 100
        cnvg_rbc_avg_per = np.average(cnvg_rbc[np.isfinite(cnvg_rbc)])  # np.abs(np.average(cnvg_rbc_old) - np.average(cnvg_rbc)) / np.average(cnvg_rbc_old) * 100
        cnvg_rbc_max_per = np.max(cnvg_rbc[np.isfinite(cnvg_rbc)])  # np.abs(np.max(cnvg_rbc_old) - np.max(cnvg_rbc)) / np.max(cnvg_rbc_old) * 100

        # find the position of the max value
        # max_position_flow, _ = max(enumerate(flow_mask), key=lambda x: x[1])

        flownetwork.iteration += 1

        if cnvg_flow_avg_per < 0.5 and cnvg_flow_max_per < 1 and cnvg_rbc_avg_per < 0.5 and cnvg_rbc_max_per < 1:
            flownetwork.convergence_check = True
            print(f"Iteration number {flownetwork.iteration}.2e and data: {cnvg_flow_avg_per, cnvg_flow_max_per, cnvg_rbc_avg_per, cnvg_rbc_max_per}")
        else:
            flownetwork.convergence_check = False
            vessel = 10478
            specific_vessel = np.abs(old_flow[vessel] - flow_rate[vessel]) / old_flow[vessel] * 100
            save_data = np.append(save_data, specific_vessel)  # , flow_rate[vessel])

            if flownetwork.iteration % 10 == 0:
                with open(PARAMETERS['path_output_file'], 'a') as file:
                    file.write(f"Itr: {flownetwork.iteration}   rbc_mean(%):{cnvg_rbc_avg_per:.2e} rbc_max(%):{cnvg_rbc_max_per :.2e} flow_mean(%): {cnvg_flow_avg_per:.2e} "
                               f"flow_max(%):{cnvg_flow_max_per:.2e}  n_es_max_: {position_array_mask[key_cnvg_flow_max]} old_flow:"
                               f"{old_flow[position_array_mask[key_cnvg_flow_max]]:.2e} crnt_flow:{flow_rate[position_array_mask[key_cnvg_flow_max]]:.2e} "
                               f" {np.count_nonzero(cnvg_flow_per >= 1)}(1%) {np.count_nonzero(cnvg_flow_per >= 5)}(5%) {np.count_nonzero(cnvg_flow_per >= 10)}(10%)"
                               f"{np.count_nonzero(cnvg_flow_per >= 50)}(50%) {np.count_nonzero(cnvg_flow_per >= 100)}(100%) \n")

                # util_convergence_plot(flownetwork, save_data, PARAMETERS, f" error % for vessel {vessel}:   actual flow_rate in the vessel {flow_rate[vessel]:.2e} \n min "
                #                                                           f"of all "
                #                                                           f"vessels:{min(flow_rate[flow_rate != 0]):.2e} max of all vessels: {max(flow_rate):.2e}")
                # print(f"max than 1% {np.count_nonzero(cnvg_flow_per >= 1)}")
                # print(f"max than 5% {np.count_nonzero(cnvg_flow_per >= 5)}")
                # print(f"max than 10% {np.count_nonzero(cnvg_flow_per >= 10)}")
                # print(f"max than 50% {np.count_nonzero(cnvg_flow_per >= 50)}")
                # print(f"max than 100% {np.count_nonzero(cnvg_flow_per >= 100)}")
            # print(
            #     f"Iteration number {flownetwork.iteration} and max error: "
            #     f"{cnvg_flow_max_per:.2e}  old_flow:{old_flow[position_array_mask[key_cnvg_flow_max]]:.2e}  "
            #     f"current_flow:{flow_rate[position_array_mask[key_cnvg_flow_max]]:.2e}  "
            #     f"mean: {np.abs(old_flow[position_array_mask[key_cnvg_flow_max]] - flow_rate[position_array_mask[key_cnvg_flow_max]]):.2e}"
            #     f" pos_orig: {position_array_mask[key_cnvg_flow_max]}   ")  # more than 100%

    print("Convergence: DONE in -> " + str(flownetwork.iteration))

    # s_curve_util(PARAMETERS, flownetwork)
    #
    # s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.1)
    # s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.3)
    # s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.5)
    # s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.7)
    #
    # s_curve_util_trifurcation(PARAMETERS, flownetwork)
