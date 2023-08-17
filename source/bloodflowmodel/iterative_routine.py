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
    save_data_max_flow, save_data_max_rbc, save_data_max_hemat = [], [], []
    save_data_avg_flow, save_data_avg_rbc, save_data_avg_hemat = [], [], []
    # save_data = np.append(save_data, flownetwork.flow_rate[1876])

    with open(PARAMETERS['path_output_file'], 'a') as file:
        file.write(f"Network: {PARAMETERS['network_name']} \nnr of vs: {flownetwork.nr_of_vs} - nr of boundary vs: {len(flownetwork.boundary_vs)} - nr of es:"
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
        position_array_mask = np.arange(len(cnvg_flow_per))[np.isfinite(cnvg_flow_per)]

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

        flownetwork.iteration += 1

        if cnvg_flow_avg_per < 0.5 and cnvg_flow_max_per < 1 and cnvg_rbc_avg_per < 0.5 and cnvg_rbc_max_per < 1:
            print(f"Iteration number {flownetwork.iteration} and "
                  f"FLOW {cnvg_flow_avg_per:.2e} {cnvg_flow_max_per:.2e}  RBC {cnvg_rbc_avg_per:.2e} {cnvg_rbc_max_per:.2e}  HEMATOCRIT {cnvg_hem_avg_per:.2e}"
                  f" {cnvg_hem_max_per:.2e}")

            util_convergence_plot(flownetwork, save_data_max_flow, PARAMETERS, f" flow error % max for all vessel")
            util_convergence_plot(flownetwork, save_data_max_rbc, PARAMETERS, f" rbc error % max for all vessel")
            util_convergence_plot(flownetwork, save_data_max_hemat, PARAMETERS, f" hemat error % max for all vessel")

            flownetwork.convergence_check = True

        else:

            if flownetwork.iteration % 100 == 0:
                util_convergence_plot(flownetwork, save_data_max_flow[-100:], PARAMETERS, f" flow error % max for all vessel", "max/flow_max")
                util_convergence_plot(flownetwork, save_data_max_rbc[-100:], PARAMETERS, f" rbc error % max for all vessel", "max/rbc_max")
                util_convergence_plot(flownetwork, save_data_max_hemat[-100:], PARAMETERS, f" hemat error % max for all vessel", "max/hemat_max")
                util_convergence_plot(flownetwork, save_data_max_flow[-100:], PARAMETERS, f" flow error % max for all vessel", "average/flow_avg")
                util_convergence_plot(flownetwork, save_data_max_rbc[-100:], PARAMETERS, f" rbc error % max for all vessel", "average/rbc_avg")
                util_convergence_plot(flownetwork, save_data_max_hemat[-100:], PARAMETERS, f" hemat error % max for all vessel", "average/hemat_avg")
                with open(PARAMETERS['path_output_file'], 'a') as file:
                    file.write(f"Itr {flownetwork.iteration} and data: "
                               f"FLOW {cnvg_flow_avg_per:.2e} {cnvg_flow_max_per:.2e}  RBC {cnvg_rbc_avg_per:.2e} {cnvg_rbc_max_per:.2e}  HEMATOCRIT {cnvg_hem_avg_per:.2e}"
                               f" {cnvg_hem_max_per:.2e} {np.count_nonzero(cnvg_flow_per >= 1)}(1%) {np.count_nonzero(cnvg_flow_per >= 5)}(5%) "
                               f"{np.count_nonzero(cnvg_flow_per >= 10)}(10%) "
                               f"{np.count_nonzero(cnvg_flow_per >= 50)}(50%) {np.count_nonzero(cnvg_flow_per >= 100)}(100%) \n")

            flownetwork.convergence_check = False

    print("Convergence: DONE in -> " + str(flownetwork.iteration))

    s_curve_util(PARAMETERS, flownetwork)

    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.1)
    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.3)
    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.5)
    s_curve_personalized_thersholds(flownetwork, PARAMETERS, 0.7)

    s_curve_util_trifurcation(PARAMETERS, flownetwork)
