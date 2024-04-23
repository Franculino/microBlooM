import os
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from math import e
from matplotlib.ticker import FuncFormatter
from numpy import arange
import seaborn as sns


# FUNCTION TO ASSIST THE PLOTS
def _logit(x):
    return np.log(x / (1 - x))


def s_curve(hemat_par, fractional_flow_a, diam_par, diam_a, diam_b, qRBCp):
    x_o_init = 0.964  # micrometers
    A_o_init = -13.29  # micrometers
    B_o_init = 6.98  # micrometers

    diam_a, diam_b, diam_par = diam_a * 1E6, diam_b * 1E6, diam_par * 1E6

    flow_parent = hemat_par * qRBCp
    flow_a = fractional_flow_a / flow_parent
    fractional_flow_b = 1 - fractional_flow_a
    flow_b = fractional_flow_b / flow_parent

    x_0 = x_o_init * (1 - hemat_par) / diam_par

    A = A_o_init * ((pow(diam_a, 2) - pow(diam_b, 2)) / (pow(diam_a, 2) + pow(diam_b, 2))) * (1 - hemat_par) / diam_par
    # A = A_o_init * ((diam_a - diam_b) / (diam_a + diam_b)) * ((1 - hemat_par) / diam_par)
    # A = A_o_init * np.log(diam_a / diam_b) * (1/ diam_par)
    B = 1 + (B_o_init * (1 - hemat_par)) / diam_par

    if fractional_flow_a <= x_0:
        fractional_qRBCa, fractional_qRBCb = 0, 1
    elif x_0 < fractional_flow_a < (1 - x_0):
        logit_F_Q_a_e = A + B * _logit((fractional_flow_a - x_0) / (1 - (2 * x_0)))
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


def save_fig(path, title):
    """
    Create a new directory id doesn't exist and save the graph if needed

    @param flownetwork:
    @type flownetwork:
    @param path: address to store the graph
    @type path: string
    """
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    plt.savefig(path + title + '.png')


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


def quartile_assistant(data):
    #  data = np.log10(data)

    # Calculate the 25th and 75th percentiles
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate the Interquartile Range (IQR)
    IQR = 1.5 * (Q3 - Q1)

    # # Define the lower and upper bounds for outliers
    lower_bound = Q1 - IQR
    upper_bound = Q3 + IQR

    # inside interval
    data = data[data > lower_bound]
    data = data[data < upper_bound]
    # data = data
    # val True = flow
    # val False = hd
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)

    return data, mean_val, median_val, std_val, Q1, Q3


def format_y_ticks(value, _):
    return f'$10^{{{int(value)}}}$'


# PLOTS
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
        save_fig(path=PARAMETERS['path_for_graph'] + '/iteration_graph/trifurc')
    plt.close()


def s_curve_util(PARAMETERS, flownetwork):
    lista, list_b, listc, listd, liste, listf, listg, listh, listx, listz, cc, dd, ee, ff, gg, hh, ii, kk, zz, xx = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
        [], [], [], [], []
    plt.figure(figsize=(13, 13), dpi=200)
    plt.style.use('seaborn-whitegrid')

    parent, Da, Db = 5E-6, 4E-6, 5E-6
    title = f""  # "S-Curve Dp= {parent:.2e} Da= {Da} Db= {Db}"

    for i in arange(0, 1, 0.01):
        a, b, c, d = s_curve(0.1, i, parent, Da, Db, flownetwork.hd[0] * flownetwork.flow_rate[0])
        lista.append(a)
        list_b.append(b)
        cc.append(c)
        dd.append(d)
    plt.plot(dd, cc, 'r')
    plt.plot(list_b, lista, 'r')

    for i in arange(0, 1, 0.01):
        a, b, c, d = s_curve(0.3, i, parent, Da, Db, flownetwork.hd[0] * flownetwork.flow_rate[0])
        listc.append(a)
        listd.append(b)
        ee.append(c)
        ff.append(d)
    plt.plot(ff, ee, 'b')
    plt.plot(listd, listc, 'b')

    for i in arange(0, 1, 0.01):
        a, b, c, d = s_curve(0.5, i, parent, Da, Db, flownetwork.hd[0] * flownetwork.flow_rate[0])
        liste.append(a)
        listf.append(b)
        gg.append(c)
        hh.append(d)
    plt.plot(hh, gg, 'g')
    plt.plot(listf, liste, 'g')

    for i in arange(0, 1, 0.01):
        a, b, c, d = s_curve(0.7, i, parent, Da, Db, flownetwork.hd[0] * flownetwork.flow_rate[0])
        listg.append(a)
        listh.append(b)
        ii.append(c)
        kk.append(d)
    plt.plot(kk, ii, 'm')
    plt.plot(listh, listg, 'm')

    for i in arange(0, 1, 0.01):
        a, b, c, d = s_curve(0.9, i, parent, Da, Db, flownetwork.hd[0] * flownetwork.flow_rate[0])
        listx.append(a)
        listz.append(b)
        xx.append(c)
        zz.append(d)
    plt.plot(zz, xx, 'k')
    plt.plot(listz, listx, 'k')

    plt.plot(flownetwork.fractional_a_blood, flownetwork.fractional_a_qRBCs, "o", markersize=3, color='olive')
    plt.plot(flownetwork.fractional_b_blood, flownetwork.fractional_b_qRBCs, "o", markersize=3, color='aquamarine')

    plt.title(title, fontsize=25)
    plt.xlabel("Fractional Blood Flow", fontsize=20)
    plt.ylabel("Fractional Erythrocity Flow", fontsize=20)
    plt.legend(["HD:0.1", "HD:0.1 ", "HD:0.3", "HD:0.3", "HD:0.5", "HD:0.5", "HD:0.7", "HD:0.7", "HD:0.9", "HD:0.9"], fontsize=20)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.20), fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.20), fontsize=20)

    if PARAMETERS['save']:
        save_fig(path=PARAMETERS['path_for_graph'] + '/s_curve/' + title)

    plt.close()


def s_curve_personalized_thersholds(flownetwork, interval):
    lista, list_b, listc, listd, liste, listf, listg, listh, listx, listz, cc, dd, ee, ff, gg, hh, ii, kk, xx, zz = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    plt.figure(figsize=(13, 13), dpi=200)
    plt.style.use('seaborn-whitegrid')
    threshold = 0.05
    i = 0
    for element in flownetwork.hemat_parent_plot:
        if (interval - threshold) <= element <= (interval + threshold):
            plt.plot(flownetwork.fractional_a_blood[i], flownetwork.fractional_a_qRBCs[i], "o", markersize=3)
            plt.plot(flownetwork.fractional_b_blood[i], flownetwork.fractional_b_qRBCs[i], "o", markersize=3)
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

        case 0.9:
            for i in arange(0, 1, 0.01):
                a, b, c, d = s_curve(0.9, i, 5E-6, 5E-6, 5E-6, flownetwork.hd[0] * flownetwork.flow_rate[0])
                listx.append(a)
                listz.append(b)
                xx.append(c)
                zz.append(d)
            plt.plot(zz, xx, 'k')
            plt.plot(listz, listx, 'k')
            # plt.legend(["daughter a", " daughter b", "HD:0.7"])

    plt.style.use('seaborn-whitegrid')
    plt.title("S-Curve refer to parent hematocrit of " + str(interval) + " in a interval of ± " + str(threshold))
    plt.xlabel("Fractional Blood Flow")
    plt.ylabel("Fractional Erythrocyte Flow")

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.20))
    plt.yticks(np.arange(0, 1.1, 0.20))

    # if flownetwork._PARAMETERS['save']:
    #     save_fig(path=flownetwork._PARAMETERS['path_for_graph'] + '/iteration_graph/s_curve_interval_near_' + str(interval))
    plt.close()


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


def util_display_graph(g, PARAMETERS):
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

    # if flownetwork._PARAMETERS['save']:
    #     save_fig(path=PARAMETERS['path_for_graph'] + '/iteration_graph/' + str(PARAMETERS['boundary_hematocrit']) + '/' + str(flownetwork.iteration) + '_HD_'
    #     + str(PARAMETERS['boundary_hematocrit']))
    plt.close()


def util_convergence_plot(flownetwork, iteration_plot, PARAMETERS, title, path_title, name):
    plt.figure(figsize=(15, 15), dpi=300)

    plt.style.use('seaborn-whitegrid')
    # title = "Convergence plot after " + str(flownetwork.iteration) + " iteration"
    plt.plot(range((flownetwork.iteration - 100), flownetwork.iteration), iteration_plot, "-")  # , '-ok')
    plt.title(title, fontsize=10)
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Error difference in %", fontsize=10)
    plt.yscale("log")
    plt.xticks(range((flownetwork.iteration - 100), flownetwork.iteration, 10), fontsize=20)
    plt.yticks(fontsize=20)

    if PARAMETERS['save']:
        save_fig(path=PARAMETERS['path_for_graph'] + '/iteration_graph/' + PARAMETERS["network_name"] + '/' + path_title + name + '/' + name + '_cnv_' + str(
            flownetwork.iteration))
    plt.close()


def percentage_vessel_plot(flownetwork, iteration_plot, PARAMETERS, title, path_title):
    plt.figure(figsize=(15, 15), dpi=300)

    plt.style.use('seaborn-whitegrid')
    # title = "Convergence plot after " + str(flownetwork.iteration) + " iteration"
    plt.plot(range(0, flownetwork.iteration), iteration_plot, "-")  # , '-ok')
    plt.title(title, fontsize=10)
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("% of vessel", fontsize=10)
    plt.xticks(range(0, flownetwork.iteration, 10), fontsize=20)
    plt.yticks(fontsize=20)

    if PARAMETERS['save']:
        save_fig(path=PARAMETERS['path_for_graph'] + '/convergence_vessel/' + PARAMETERS["network_name"] + '/' + path_title + '/' + str(
            flownetwork.iteration) + '_' + path_title)
    plt.close()


def util_convergence_plot_final(flownetwork, iteration_plot, PARAMETERS, title, path_title, name):
    plt.figure(figsize=(15, 15), dpi=300)

    plt.style.use('seaborn-whitegrid')
    plt.plot(range(0, flownetwork.iteration), iteration_plot, "-")  # , '-ok')
    plt.title(title, fontsize=10)
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Error difference in %", fontsize=10)
    plt.yscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if PARAMETERS['save']:
        save_fig(path=PARAMETERS['path_for_graph'] + '/iteration_graph/' + PARAMETERS["network_name"] + '/' + path_title + name + '_cnv_' + str(
            flownetwork.iteration))
    plt.close()


def residual_plot(flownetwork, PARAMETERS, title, path_title, name):
    plt.figure(figsize=(15, 15), dpi=300)
    plt.style.use('seaborn-whitegrid')

    # Plot lines with labels
    plt.plot(range(0, len(flownetwork.residualOverIterationMax)), flownetwork.residualOverIterationMax, "-k", label="residualMax")
    plt.plot(range(0, len(flownetwork.residualOverIterationNorm)), flownetwork.residualOverIterationNorm, "-g", label="residualMean")
    x_threshold = np.linspace(0, 10, len(flownetwork.residualOverIterationMax))

    plt.plot(range(0, len(flownetwork.residualOverIterationMax)), np.full_like(x_threshold, flownetwork.two_MagnitudeThreshold), color='b', linestyle=':',
             label="2MagnitudeThreshold")
    plt.xlim([0, len(flownetwork.residualOverIterationMax)])
    plt.title(title)  # , fontsize=10)
    plt.xlabel("Iteration")  # , fontsize=10)
    plt.ylabel("Residual value")  # , fontsize=10)
    plt.yscale("log")
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)+

    # plt.ylim(None, 1e-11)
    plt.xlim([0, len(flownetwork.residualOverIterationMax)])

    plt.rcParams.update({'font.size': 22})

    # To display Alphas
    # x = range(0, flownetwork.iteration)
    # for i, label in enumerate(flownetwork.alphaSave):
    #     x_label = x[i * 50] if i * 50 < len(x) else x[-1]
    #     y_label = residualNorm[i * 50] if i * 50 < len(residualNorm) else residualNorm[-1]
    #     plt.scatter(x_label, y_label, color='red', marker='o')  # Include label parameter here
    #     plt.annotate(label, (x_label, y_label), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=20)

    # Create the legend
    plt.legend(loc='upper right', fontsize=20)

    if PARAMETERS['save']:
        save_fig(path=PARAMETERS['path_for_graph'] + '/' + path_title, title=name + '/' + str(flownetwork.iteration))
    plt.close()


def residual_plot_berg(flownetwork, PARAMETERS, title, path_title, name):
    plt.figure(figsize=(15, 15), dpi=300)
    plt.style.use('seaborn-whitegrid')
    iteration = flownetwork.iteration + 1
    # Plot lines with labels
    plt.plot(range(0, iteration), flownetwork.bergIteration, "-k", label="Residual Berg")
    x_threshold = np.linspace(0, 10, iteration)
    plt.plot(range(0, iteration), np.full_like(x_threshold, flownetwork.berg_criteria), color='r', linestyle=':', label="Threshold")

    plt.xlim([0, iteration])
    plt.title(title)  # , fontsize=10)
    plt.xlabel("Iteration")  # , fontsize=10)
    plt.ylabel("Residual Berg value")  # , fontsize=10)
    plt.yscale("log")
    # plt.ylim(None, 1e-11)
    plt.rcParams.update({'font.size': 22})

    plt.legend(loc='upper right', fontsize=20)

    if PARAMETERS['save']:
        save_fig(path=PARAMETERS['path_for_graph'] + '/' + path_title + name + '/' + str(flownetwork.iteration))
    plt.close()


def residual_plot_berg_subset(flownetwork, residualBerg, PARAMETERS, title, path_title, name, label):
    plt.figure(figsize=(15, 15), dpi=300)
    plt.style.use('seaborn-whitegrid')

    # Plot lines with labels
    plt.plot(range(0, flownetwork.iteration), residualBerg, "-k", label=label)
    x_threshold = np.linspace(0, 10, flownetwork.iteration)
    plt.plot(range(0, flownetwork.iteration), np.full_like(x_threshold, flownetwork.berg_criteria), color='r', linestyle=':', label="Threshold")

    plt.xlim([0, flownetwork.iteration])
    plt.title(title)  # , fontsize=10)
    plt.xlabel("Iteration")  # , fontsize=10)
    plt.ylabel("Residual Berg value")  # , fontsize=10)
    plt.yscale("log")
    # plt.ylim(None, 1e-11)
    plt.rcParams.update({'font.size': 22})

    plt.legend(loc='upper right', fontsize=20)

    if PARAMETERS['save']:
        save_fig(path=PARAMETERS['path_for_graph'] + '/' + path_title + name + '/' + str(flownetwork.iteration))
    plt.close()


def residual_plot_rasmussen(flownetwork, PARAMETERS, title, path_title, name):
    plt.figure(figsize=(15, 15), dpi=300)
    plt.style.use('seaborn-whitegrid')
    iter = flownetwork.iteration + 1
    # Plot lines with labels
    plt.plot(range(0, iter), flownetwork.hd_convergence_criteria_plot, color="lightcoral", label="Hematocrit")
    plt.plot(range(0, iter), flownetwork.flow_convergence_criteria_plot, color="cornflowerblue", label="Flow")
    x_threshold = np.linspace(0, 10, iter)
    plt.plot(range(0, iter), np.full_like(x_threshold, flownetwork.hd_convergence_criteria_plot), color='indianred', linestyle=':',
             label="Threshold Hematocrit", linewidth=5)
    x_threshold = np.linspace(0, 10, iter)
    plt.plot(range(0, iter), np.full_like(x_threshold, flownetwork.flow_convergence_criteria_plot), color='royalblue', linestyle=':', label="Threshold Flow",
             linewidth=5)
    plt.xlim([1, iter])
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Change between iterations")
    plt.yscale("log")
    plt.rcParams.update({'font.size': 22})

    plt.legend(loc='upper right', fontsize=20)

    if PARAMETERS['save']:
        save_fig(path=PARAMETERS['path_for_graph'] + '/' + path_title + name + '/' + str(flownetwork.iteration))
    plt.close()


def residual_graph(data, PARAMETERS, title, name):
    plt.figure(figsize=(30, 30), dpi=300)
    plt.style.use('seaborn-whitegrid')
    # Plot lines with labels for the last 100 iterations
    plt.plot(range(0, len(data)), data, "-k", label=str(name))
    plt.title(name + " " + str(title))  # , fontsize=30)
    plt.xlabel("Iteration")  # , fontsize=10)
    plt.ylabel(name)
    plt.ticklabel_format(useOffset=False)
    plt.rcParams.update({'font.size': 22})
    # Create the legend
    plt.legend(loc='upper right')  # , fontsize=20)

    if PARAMETERS['save']:
        save_fig(path=PARAMETERS['path_for_graph'] + '/' + name + '/' + str(title))

    plt.close()


def residual_plot_last_iteration(flownetwork, residualMax, residualNorm, PARAMETERS, title, path_title, name):
    plt.figure(figsize=(15, 15), dpi=300)
    plt.style.use('seaborn-whitegrid')
    # Calculate the starting iteration for the last 100 iterations
    start_iteration = max(0, flownetwork.iteration - 300)

    # Plot lines with labels for the last 100 iterations
    plt.plot(range(start_iteration, flownetwork.iteration), residualMax[start_iteration:flownetwork.iteration], "-k", label="residualMax")
    plt.plot(range(start_iteration, flownetwork.iteration), residualNorm[start_iteration:flownetwork.iteration], "-g", label="residualMean")
    x_threshold = np.linspace(start_iteration, flownetwork.iteration, flownetwork.iteration - start_iteration)
    plt.plot(range(start_iteration, flownetwork.iteration), np.full_like(x_threshold, flownetwork.zeroFlowThreshold), color='r', linestyle=':',
             label="zeroFlowThreshold")

    plt.title(title, fontsize=10)
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Residual value", fontsize=10)
    plt.yscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(None, 1e-11)

    x = range(start_iteration, flownetwork.iteration)
    for i, label in enumerate(flownetwork.alphaSave[start_iteration:flownetwork.iteration]):
        x_label = x[i * 50] if i * 50 < len(x) else x[-1]
        y_label = residualNorm[start_iteration + i * 50] if start_iteration + i * 50 < len(residualNorm) else residualNorm[-1]
        plt.scatter(x_label, y_label, color='red', marker='o')  # Include label parameter here
        plt.annotate(label, (x_label, y_label), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=20)

    # Create the legend
    plt.legend(loc='upper right', fontsize=20)

    if PARAMETERS['save']:
        save_fig(path=PARAMETERS['path_for_graph'] + '/' + PARAMETERS["network_name"] + '/' + path_title + name + '/' + str(flownetwork.iteration))

    plt.close()


def frequency_plot_abs_fl(PARAMETERS, data, title, it, path_plot):
    plt.figure(figsize=(15, 7.5), dpi=300)

    data, mean_val, median_val, std_val, Q1, Q3 = quartile_assistant(data)
    sns.histplot(data, bins='auto', kde=False, edgecolor='black', stat="percent", label='Our', color=sns.color_palette(palette='Blues')[2],
                 multiple="stack")
    plt.axvline(mean_val, color='red', linestyle='dashed', label='Mean', linewidth=1)
    plt.axvline(median_val, color='blue', linestyle='dashed', label='Median', linewidth=1)
    plt.axvline(mean_val + std_val, color='green', linestyle='dashed', label='Mean + Std', linewidth=1)
    plt.axvline(mean_val - std_val, color='orange', linestyle='dashed', label='Mean - Std', linewidth=1)

    plt.legend(loc='upper right')
    plt.title('Our - ' + title + str(it), fontsize=14)
    plt.xlabel('Magnitude', fontsize=12)
    plt.ylabel('Percentage(%)', fontsize=12)
    plt.xticks(fontsize=12)
    # plt.gca().xaxis.set_major_formatter(FuncFormatter(format_y_ticks))

    # Adjust layout
    plt.tight_layout()
    # Show the plot

    if PARAMETERS['save']:
        save_fig(path=path_plot, title='frequency_flow_' + title + '_' + str(it))


def relative_difference(flownetwork, data_currentFlow, data_convergedFlow, data_currentHD, data_convergedHD, two_MagnitudeThreshold, path_plot):
    # BERG
    plt.figure(figsize=(15, 7.5), dpi=300)
    plt.subplot(1, 2, 1)
    reference_difference_Flow = abs(data_currentFlow - data_convergedFlow) / data_currentFlow
    # Rimuovi i valori NaN
    reference_difference_Flow = reference_difference_Flow[np.isfinite(reference_difference_Flow)]

    mean_val, median_val, std_val = np.mean(reference_difference_Flow), np.median(reference_difference_Flow), np.std(
        reference_difference_Flow[reference_difference_Flow != 0])

    sns.histplot(reference_difference_Flow[reference_difference_Flow != 0], bins='auto', kde=False, edgecolor='black', stat="percent",
                 log_scale=True, label='Flow',
                 color=sns.color_palette(palette='Blues')[3])

    plt.axvline(mean_val, color='red', linestyle='dashed', label='Mean', linewidth=1)
    plt.axvline(median_val, color='blue', linestyle='dashed', label='Median', linewidth=1)
    plt.axvline(two_MagnitudeThreshold, color='violet', linestyle='dashed', label='two_MagnitudeThreshold', linewidth=1)
    plt.xlim([1e-10, 1e0])

    plt.legend(loc='upper right')
    plt.title(f'Relative Difference Flow at iteration {flownetwork.iteration} - Converged')
    # Adjust layout
    plt.tight_layout()

    plt.subplot(1, 2, 2)

    reference_difference_HD = abs(data_currentHD - data_convergedHD) / data_currentHD
    # Rimuovi i valori NaN
    reference_difference_HD = reference_difference_HD[np.isfinite(reference_difference_HD)]

    mean_val, median_val, std_val = np.mean(reference_difference_HD), np.median(reference_difference_HD), np.std(
        reference_difference_HD[reference_difference_HD != 0])

    sns.histplot(reference_difference_HD[reference_difference_HD != 0], bins='auto', kde=False, edgecolor='black', stat="percent",
                 log_scale=True, label='HD',
                 color=sns.color_palette(palette='rocket')[3])

    plt.axvline(mean_val, color='red', linestyle='dashed', label='Mean', linewidth=1)
    plt.axvline(median_val, color='blue', linestyle='dashed', label='Median', linewidth=1)
    plt.legend(loc='upper right')
    plt.title(f'Relative Difference HD at iteration {flownetwork.iteration} - Converged')
    # Adjust layout
    plt.tight_layout()
    plt.xlim([1e-10, 1e0])

    if flownetwork._PARAMETERS['save']:
        save_fig(path=path_plot, title='relative_difference_ ' + str(flownetwork.iteration) + '.png')
