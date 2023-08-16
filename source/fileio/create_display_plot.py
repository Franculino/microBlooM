import igraph as ig
from igraph import Graph, Plot

import matplotlib.pyplot as plt
import numpy as np
from math import e
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from numpy import arange
import seaborn as sns


def _logit(x):
    return np.log(x / (1 - x))


def s_curve(hemat_par, fractional_flow_a, diam_par, diam_a, diam_b, qRBCp):
    x_o_init = 1.12  # micrometers
    A_o_init = 15.47  # micrometers
    B_o_init = 8.13  # micrometers

    diam_a, diam_b, diam_par = diam_a * 1E6, diam_b * 1E6, diam_par * 1E6

    flow_parent = hemat_par * qRBCp
    flow_a = fractional_flow_a / flow_parent
    fractional_flow_b = 1 - fractional_flow_a
    flow_b = fractional_flow_b / flow_parent

    x_0 = x_o_init * (1 - hemat_par) / diam_par

    A = A_o_init * ((pow(diam_a, 2) - pow(diam_b, 2)) / (pow(diam_a, 2) + pow(diam_b, 2))) * (
            1 - hemat_par) / diam_par

    B = 1 + B_o_init * (1 - hemat_par) / diam_par

    if fractional_flow_a <= x_0:
        fractional_qRBCa, fractional_qRBCb = 0, 1
    elif x_0 < fractional_flow_a < (1 - x_0):
        logit_F_Q_a_e = A + B * _logit((fractional_flow_a - x_0) / (1 - x_0))
        fractional_qRBCa = (pow(e, logit_F_Q_a_e) / (1 + pow(e, logit_F_Q_a_e)))
        fractional_qRBCb = 1 - fractional_qRBCa
    elif fractional_flow_a >= (1 - x_0):
        fractional_qRBCa, fractional_qRBCb = 1, 0

    if fractional_qRBCa == 0:
        hemat_a = 0
        hemat_b = (fractional_qRBCb * qRBCp) / flow_b
    elif fractional_qRBCb == 0:
        hemat_a = (fractional_qRBCa * qRBCp) / flow_a
        hemat_b = 0
    else:
        hemat_a = (fractional_qRBCa * qRBCp) / flow_a
        hemat_b = (fractional_qRBCb * qRBCp) / flow_b

    threshold = 0.99
    if hemat_b >= threshold:
        hemat_surplus = hemat_b - threshold
        fractional_RBCs_suprlus = (hemat_surplus * flow_b) / qRBCp
        fractional_qRBCb = fractional_qRBCb - fractional_RBCs_suprlus
        fractional_qRBCa = fractional_qRBCa + fractional_RBCs_suprlus
    elif hemat_a >= threshold:
        hemat_surplus = hemat_a - threshold
        fractional_RBCs_suprlus = (hemat_surplus * flow_a) / qRBCp
        fractional_qRBCb = fractional_qRBCb + fractional_RBCs_suprlus
        fractional_qRBCa = fractional_qRBCa - fractional_RBCs_suprlus

    return fractional_qRBCa, fractional_flow_a, fractional_qRBCb, fractional_flow_b


def s_curve_util_trifurcation(PARAMETERS, flownetwork):
    plt.figure(figsize=(13, 13), dpi=200)
    plt.style.use('seaborn-whitegrid')
    # print(flownetwork.fractional_trifurc_blood)
    i = 0
    while i < len(flownetwork.fractional_trifurc_blood):
        plt.plot(flownetwork.fractional_trifurc_blood[i:i + 3], flownetwork.fractional_trifurc_RBCs[i:i + 3], "o")
        # print(flownetwork.fractional_trifurc_blood[i:i + 3])
        i += 3

    plt.plot([0, 0.5, 1], [0, 0.5, 1], 'black')
    plt.title("Fractional bulk vs fractional RBC flow in Trifurcation Case")
    plt.xlabel("fractional bulk flow")
    plt.ylabel("fractional RBC flow")
    # plt.legend()
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.20))
    plt.yticks(np.arange(0, 1.1, 0.20))

    if PARAMETERS['save']:
        plt.savefig(
            PARAMETERS['path_for_graph'] + '/iteration_graph/trifurc.png')
    plt.show()
    # pass



def s_curve_util(PARAMETERS, flownetwork):
    lista, list_b, listc, listd, liste, listf, listg, listh, cc, dd, ee, ff, gg, hh, ii, kk = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    plt.figure(figsize=(13, 13), dpi=200)
    plt.style.use('seaborn-whitegrid')
    for i in arange(0, 1, 0.01):
        a, b, c, d = s_curve(0.1, i, 5E-6, 5E-6, 5E-6, flownetwork.hd[0] * flownetwork.flow_rate[0])
        lista.append(a)
        list_b.append(b)
        cc.append(c)
        dd.append(d)
    plt.plot(dd, cc, 'r')
    plt.plot(list_b, lista, 'r')

    for i in arange(0, 1, 0.01):
        a, b, c, d = s_curve(0.3, i, 5E-6, 5E-6, 5E-6, flownetwork.hd[0] * flownetwork.flow_rate[0])
        listc.append(a)
        listd.append(b)
        ee.append(c)
        ff.append(d)
    plt.plot(ff, ee, 'b')
    plt.plot(listd, listc, 'b')

    for i in arange(0, 1, 0.01):
        a, b, c, d = s_curve(0.5, i, 5E-6, 5E-6, 5E-6, flownetwork.hd[0] * flownetwork.flow_rate[0])
        liste.append(a)
        listf.append(b)
        gg.append(c)
        hh.append(d)
    plt.plot(hh, gg, 'g')
    plt.plot(listf, liste, 'g')

    for i in arange(0, 1, 0.01):
        a, b, c, d = s_curve(0.7, i, 5E-6, 5E-6, 5E-6, flownetwork.hd[0] * flownetwork.flow_rate[0])
        listg.append(a)
        listh.append(b)
        ii.append(c)
        kk.append(d)
    plt.plot(kk, ii, 'm')
    plt.plot(listh, listg, 'm')

    plt.plot(flownetwork.fractional_a_blood, flownetwork.fractional_a_qRBCs, "o")
    plt.plot(flownetwork.fractional_b_blood, flownetwork.fractional_b_qRBCs, "o")

    plt.title("S-Curve ")
    plt.xlabel("fractional blood flow")
    plt.ylabel("fractional erythrocity flow")
    plt.legend(["HD:0.1", "HD:0.1 ", "HD:0.3", "HD:0.3", "HD:0.5", "HD:0.5", "HD:0.7", "HD:0.7"])
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.20))
    plt.yticks(np.arange(0, 1.1, 0.20))

    if PARAMETERS['save']:
        plt.savefig(
            PARAMETERS['path_for_graph'] + '/iteration_graph/s_curve.png')
    plt.show()



def s_curve_personalized_thersholds(flownetwork, PARAMETERS, interval):
    lista, list_b, listc, listd, liste, listf, listg, listh, cc, dd, ee, ff, gg, hh, ii, kk = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    plt.figure(figsize=(13, 13), dpi=200)
    plt.style.use('seaborn-whitegrid')
    threshold = 0.1
    i = 0
    for element in flownetwork.hemat_parent_plot:
        if (interval - threshold) <= element <= (interval + threshold):
            plt.plot(flownetwork.fractional_a_blood[i], flownetwork.fractional_a_qRBCs[i], "o")
            plt.plot(flownetwork.fractional_b_blood[i], flownetwork.fractional_b_qRBCs[i], "o")
        i += 1

    match interval:
        case 0.1:
            for i in arange(0, 1, 0.01):
                a, b, c, d = s_curve(0.1, i, 5E-6, 5E-6, 5E-6, flownetwork.hd[0] * flownetwork.flow_rate[0])
                lista.append(a)
                list_b.append(b)
                cc.append(c)
                dd.append(d)
            plt.plot(dd, cc, 'r')
            plt.plot(list_b, lista, 'r')
            # plt.legend(["daughter a", " daughter b", "HD:0.1"])

        case 0.3:
            for i in arange(0, 1, 0.01):
                a, b, c, d = s_curve(0.3, i, 5E-6, 5E-6, 5E-6, flownetwork.hd[0] * flownetwork.flow_rate[0])
                listc.append(a)
                listd.append(b)
                ee.append(c)
                ff.append(d)
            plt.plot(ff, ee, 'b')
            plt.plot(listd, listc, 'b')
            # plt.legend(["daughter a", " daughter b", "HD:0.3"])

        case 0.5:
            for i in arange(0, 1, 0.01):
                a, b, c, d = s_curve(0.5, i, 5E-6, 5E-6, 5E-6, flownetwork.hd[0] * flownetwork.flow_rate[0])
                liste.append(a)
                listf.append(b)
                gg.append(c)
                hh.append(d)
            plt.plot(hh, gg, 'g')
            plt.plot(listf, liste, 'g')
            # plt.legend(["daughter a", " daughter b", "HD:0.5"])

        case 0.7:
            for i in arange(0, 1, 0.01):
                a, b, c, d = s_curve(0.7, i, 5E-6, 5E-6, 5E-6, flownetwork.hd[0] * flownetwork.flow_rate[0])
                listg.append(a)
                listh.append(b)
                ii.append(c)
                kk.append(d)
            plt.plot(kk, ii, 'm')
            plt.plot(listh, listg, 'm')
            # plt.legend(["daughter a", " daughter b", "HD:0.7"])

    plt.style.use('seaborn-whitegrid')
    plt.title("S-Curve refer to parent hematocrit of " + str(interval) + " in a interval of ± " + str(threshold))
    plt.xlabel("Fractional Blood Flow")
    plt.ylabel("Fractional Erythrocity Flow")

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.20))
    plt.yticks(np.arange(0, 1.1, 0.20))

    if PARAMETERS['save']:
        plt.savefig(
            PARAMETERS['path_for_graph'] + '/iteration_graph/s_curve_interval_near_' + str(
                interval) + '.png')
    plt.show()



def graph_creation(flownetwork):
    edge_list = flownetwork.edge_list
    graph = ig.Graph(edge_list.tolist())  # Generate igraph based on edge_list

    if flownetwork.diameter is not None:
        graph.es["diameter"] = flownetwork.diameter

    if flownetwork.length is not None:
        graph.es["length"] = flownetwork.length

    if flownetwork.flow_rate is not None:
        graph.es["flow_rate"] = flownetwork.flow_rate

    if flownetwork.rbc_velocity is not None:
        graph.es["rbc_velocity"] = flownetwork.rbc_velocity

    if flownetwork.ht is not None:
        graph.es["ht"] = flownetwork.ht

    if flownetwork.hd is not None:
        graph.es["hd"] = flownetwork.hd

    # TODO: delete this check
    if flownetwork.xyz is not None:
        graph.vs["xyz"] = flownetwork.xyz.tolist()

    if flownetwork.pressure is not None:
        graph.vs["pressure"] = flownetwork.pressure
    return graph



def util_display_graph(g, PARAMETERS, flownetwork):
    """
    Function to display the graph and give particulary color to each part
    :param g: graph created with graph_creation()
    :return: None
    :param iteration
    :param PARAMETERS
    :param flownetwork
    """

    # create the figure for the plot
    fig, ax = plt.subplots(figsize=(25, 25))
    # ax.set_title("Iteration ")  # + str(iteration))
    ig.plot(
        # graph to be shown
        g,
        bbox=[1200, 1200],
        # margin=100,
        target=ax,  # target="graph.pdf",
        # layout="auto",
        # size of the vertex
        # vertex_size=50,
        vertex_size=0.3,
        # to color the different vertex, also an example to conditionally color them in igraph
        vertex_color=["light blue" if np.isin(i, PARAMETERS["hexa_boundary_vertices"]) else "white" for i in range(0, len(g.vs["pressure"]))],
        # width of nodes outline
        vertex_frame_width=1.4,
        # color of the vertex outline
        vertex_frame_color="black",
        # nodes label
        vertex_label=[[str(round(g.vs[i]["pressure"], 3)), i] for i in range(0, len(g.vs["pressure"]))],
        # size of nodes label
        vertex_label_size=10.0,  # 20
        # size of edges
        edge_width=1,
        # color of edges
        edge_color=["steelblue" if lens == 5e-06 else "red" for lens in g.es["diameter"]],
        # edge labels
        # edge_label=[(str((g.es[i]["ht"]))) for i in  range(0, len(g.es["ht"]))],
        # edge_label=[round(num, 3) for num in g.es["ht"]] ,#str(round(g.es["ht"], 3)),
        # round(num, 1) for num in g.es["ht"]
        # edge_label= [i for i in range(0, len(g.vs))],
        edge_label=[str("Q:" + "{:.2e}".format(g.es[i]["flow_rate"])) + "\n \n" + str(
            "HD:" + "{:.2e}".format(g.es[i]["hd"])) + "\n \n" + str(i) for i in
                    range(0, len(g.es["flow_rate"]))],
        #
        # edge label size
        edge_label_size=10.0,  # 20
        edge_align_label=False,
    )
    if PARAMETERS['save']:
        #  plt.savefig( PARAMETERS['path_for_graph'] + '/iteration_graph/' + str(PARAMETERS['boundary_hematocrit']) + '/' + str( iteration) + '_HD_' + str(PARAMETERS[
        #  'boundary_hematocrit']) + '.png')
        plt.savefig(PARAMETERS['path_for_graph'] + '/iteration_graph/' + 'last_iteration.png')
    plt.show()



def util_convergence_plot(flownetwork, iteration_plot, PARAMETERS, title):
    # plt.figure(figsize=(15, 15), dpi=100)

    plt.style.use('seaborn-whitegrid')
    # title = "Convergence plot after " + str(flownetwork.iteration) + " iteration"
    plt.plot(range(0, flownetwork.iteration), iteration_plot, "-")  # , '-ok')
    plt.title(title, fontsize=10)
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Error difference", fontsize=10)
    plt.yscale("log")
    # plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)

    if PARAMETERS['save']:
        plt.savefig(
            PARAMETERS['path_for_graph'] + '/iteration_graph/convergence_plot.png')
    plt.show()


def percentage_lower_than(data, value):
    """
    Calculate the percentage of elements in the dataset that are lower than the given value.

    Parameters:
        data (numpy.ndarray): The dataset.
        value (float): The value to compare with.

    Returns:
        float: The percentage of elements lower than the given value.
    """
    lower_count = np.sum(data < value)
    total_count = len(data)
    percentage = (lower_count / total_count) * 100
    return percentage


def frequency_plot(flownetwork, data, title, x_axis, color_plot):
    mean_val = np.mean(data)
    median_val = np.median(data)
    max_val = np.max(data)

    # print("The value for " + str(title) + " of: median = " + str(median_val) + " mean = " + str(mean_val) + " min = " + str(np.min(data[data != 0])) + " max = " + str(max_val))

    percentage = percentage_lower_than(np.abs(data), flownetwork.zeroFlowThreshold)

    bin_count = 20000

    sns.histplot(data, bins=bin_count, kde=False, color=color_plot, edgecolor='white', stat="percent")

    plt.xlabel(x_axis)
    plt.ylabel('Percentage (%)')
    plt.title(title)
    sns.despine(left=True)
    plt.grid(axis='y', alpha=0.5)
    plt.xticks(fontsize=8)
    plt.tick_params(axis='y', which='both', color='#f0f0f0')

    bin_heights, _ = np.histogram(data, bins=bin_count)

    plt.axvline(mean_val, color='red', linestyle='dashed', label='Mean', linewidth=1)
    plt.axvline(median_val, color='blue', linestyle='dashed', label='Median', linewidth=1)
    plt.axvline(max_val, color='green', linestyle='dashed', label='Max', linewidth=1)
    plt.axvline(flownetwork.zeroFlowThreshold, color='orange', linestyle='dashed', label='Tolerance', linewidth=1)

    plt.xscale('log')

    plt.legend(loc='upper right')
    plt.gca().set_facecolor('#f0f0f0')
    plt.tight_layout()
    plt.show()
