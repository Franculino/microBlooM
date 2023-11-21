import copy
from abc import ABC
from collections import defaultdict
from types import MappingProxyType

import numpy as np
from abc import ABC, abstractmethod

from source.bloodflowmodel.pressure_flow_solver import set_low_flow_threshold
import sys


class FlowBalance(ABC):
    """
    Abstract base class for the implementations related to flow balance.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of FlowBalance
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def _get_flow_balance(self, flownetwork):
        """
        Abstract method to get the flow_balance
        """

    @abstractmethod
    def check_flow_balance(self, flownetwork):
        """
        Abstract method to get the check flow_balance
        """


class FlowBalanceClass(FlowBalance):

    def _get_flow_balance(self, flownetwork):

        nr_of_vs = flownetwork.nr_of_vs
        nr_of_es = flownetwork.nr_of_es

        edge_list = flownetwork.edge_list

        flow_balance = np.zeros(nr_of_vs)
        flow_rate = copy.deepcopy(flownetwork.flow_rate)

        for eid in range(nr_of_es):
            flow_balance[edge_list[eid, 0]] += flow_rate[eid]
            flow_balance[edge_list[eid, 1]] -= flow_rate[eid]

        return flow_balance

    def _get_flow_balance_rbcs(self, flownetwork):

        nr_of_vs = flownetwork.nr_of_vs
        nr_of_es = flownetwork.nr_of_es

        edge_list = flownetwork.edge_list

        flow_rbcs = np.zeros(nr_of_vs)
        flow_rate = copy.deepcopy(flownetwork.flow_rate)
        hd = copy.deepcopy(flownetwork.hd)

        for eid in range(nr_of_es):
            flow_rbcs[edge_list[eid, 0]] += np.multiply(flow_rate[eid], hd[eid])
            flow_rbcs[edge_list[eid, 1]] -= np.multiply(flow_rate[eid], hd[eid])

        return flow_rbcs

    def check_flow_balance(self, flownetwork, tol=1.00E-5):

        nr_of_vs = flownetwork.nr_of_vs
        flow_rate = flownetwork.flow_rate
        boundary_vs = flownetwork.boundary_vs
        iteration = flownetwork.iteration
        flow_balance = self._get_flow_balance(flownetwork)

        ref_flow = np.abs(flow_rate[boundary_vs[0]])
        tol_flow = tol * ref_flow

        is_inside_node = np.logical_not(np.in1d(np.arange(nr_of_vs), boundary_vs))
        positions_of_elements_not_in_boundary = np.where(is_inside_node)[0]
        local_balance = np.abs(flow_balance[is_inside_node])
        is_locally_balanced = local_balance < tol_flow

        if False in np.unique(is_locally_balanced):
            sys.exit("Is locally balanced: " + str(np.unique(is_locally_balanced)) + "(with tol " + str(tol_flow) + ")")

        balance_boundaries = flow_balance[boundary_vs]
        global_balance = np.abs(np.sum(balance_boundaries))
        is_globally_balanced = global_balance < tol_flow
        if not is_globally_balanced:
            sys.exit("Is globally balanced: " + str(is_globally_balanced) + "(with tol " + str(tol_flow) + ")")

        # zero-flow-threshold
        # The zero flow threshold is set as the max of the mass balance error for the internal nodes
        # it is computed the new flow inside
        if flownetwork._PARAMETERS["low_flow_vessel"] is True and flownetwork.zeroFlowThreshold is None:
            flownetwork.flow_rate = set_low_flow_threshold(self, flownetwork, local_balance)

        # ------------------------ other things ------------------------
        # RBC
        flow_rbcs = self._get_flow_balance_rbcs(flownetwork)
        local_balance_rbc = np.abs(flow_rbcs[is_inside_node])
        maxBalance, meanBalance = max(local_balance_rbc), np.mean(local_balance_rbc)
        flownetwork.maxBalance = maxBalance
        flownetwork.residualOverIterationMax = np.append(flownetwork.residualOverIterationMax, maxBalance)
        flownetwork.residualOverIterationNorm = np.append(flownetwork.residualOverIterationNorm, meanBalance)

        if iteration > 2:
            flownetwork.families_dict, flownetwork.vessel_general = copy.deepcopy(dict_for_families(flownetwork))

        if flownetwork.node_values_flow is None and iteration > 2:
            flownetwork.node_values = {node: [] for node in flownetwork.node_identifiers}
            flownetwork.node_values_hd = {node: [] for node in flownetwork.node_identifiers}
            flownetwork.node_values_flow = {node: [] for node in flownetwork.node_identifiers}
            flownetwork.vessel_value_hd = {vessel: [] for vessel in flownetwork.vessel_general}
            flownetwork.vessel_value_flow = {vessel: [] for vessel in flownetwork.vessel_general}

        if iteration > 2:
            flownetwork.node_relative_residual = {node: [] for node in positions_of_elements_not_in_boundary}
            flownetwork.two_MagnitudeThreshold = 1 * 10 ** (3 - flownetwork.zeroFlowThresholdMagnitude)

        if iteration == 4000:
            flownetwork.stop = True
            knoledge(self, flownetwork, local_balance_rbc, positions_of_elements_not_in_boundary)

        elif flownetwork.zeroFlowThreshold is not None and iteration > 2 and maxBalance <= flownetwork.two_MagnitudeThreshold:
            flownetwork.stop = True
            # knoledge(self, flownetwork, local_balance_rbc, positions_of_elements_not_in_boundary)

        elif iteration > 1:
            flownetwork.families_dict_total = copy.deepcopy(dict_for_families_total(flownetwork))

        flownetwork.local_balance_rbc = local_balance_rbc


def knoledge(self, flownetwork, local_balance_rbc, positions_of_elements_not_in_boundary):
    # variable needed
    vessel_general = copy.deepcopy(flownetwork.vessel_general)
    node_relative_residual, node_residual = np.zeros(flownetwork.nr_of_vs), np.zeros(flownetwork.nr_of_vs)
    # node_residual = copy.deepcopy(flownetwork.node_relative_residual)
    node_flow_change, vessel_flow_change, node_flow_change_total, vessel_flow_change_total, node_relative_residual_plot, node_residual_plot = [], [], [], [], \
        [], []
    vessel_value_hd = copy.deepcopy(flownetwork.vessel_value_hd)
    vessel_value_flow = copy.deepcopy(flownetwork.vessel_value_flow)
    flow_rate = abs(copy.deepcopy(flownetwork.flow_rate))

    families_dict_total = copy.deepcopy(dict_for_families_total(flownetwork))

    # salvo i valori di hd e flow per ogni vessel coinvolto
    for vessel in vessel_general:
        vessel_value_hd[vessel].append(flownetwork.hd[vessel])
        vessel_value_flow[vessel].append(flow_rate[vessel])
    flownetwork.vessel_value_hd, flownetwork.vessel_value_flow = vessel_value_hd, vessel_value_flow

    # valuto i nodi che effettivamente hanno un local_balance
    for node in positions_of_elements_not_in_boundary:
        # save the flow of parents and daughter for each internal nodes
        hd_par, flow_par, hd_dgs, flow_dgs = 0, 0, 0, 0
        #
        for element in families_dict_total[node]["par"]:
            hd_par += flownetwork.hd[element]
            flow_par += flow_rate[element]

        for dato in families_dict_total[node]["dgs"]:
            hd_dgs += flownetwork.hd[dato]
            flow_dgs += flow_rate[dato]

        # position of elements not in boundary è mappato su local_balance rbc
        # guardo quindi il nodo in che posizione di index è
        # quell'index sarà poi quello da usare per guardare il local_balance
        index = np.where(positions_of_elements_not_in_boundary == node)
        flow_gather = flow_par + flow_dgs
        if flow_gather != 0:
            relative_residual = local_balance_rbc[index][0] / np.average(flow_gather)
            node_relative_residual_plot.append(relative_residual)
            node_relative_residual[node] = relative_residual
        else:
            node_relative_residual_plot.append(0)
            node_relative_residual[node] = 0
        node_residual_plot.append(local_balance_rbc[index])
        node_residual[node] = local_balance_rbc[index]

    flownetwork.node_relative_residual, flownetwork.node_residual, flownetwork.positions_of_elements_not_in_boundary, flownetwork.node_relative_residual_plot, flownetwork.node_residual_plot \
        = node_relative_residual, node_residual, positions_of_elements_not_in_boundary, np.array(node_relative_residual_plot), np.array(node_residual_plot)

    # indices over the TH
    indices_over_blue, d = [], 0
    for node in positions_of_elements_not_in_boundary:
        index = np.where(positions_of_elements_not_in_boundary == node)
        if local_balance_rbc[index] > flownetwork.two_MagnitudeThreshold and local_balance_rbc[index] != 0:
            indices_over_blue.append(node)
    indices_over_blue = np.array(indices_over_blue)

    # I have the pure magnitudes for each values of the local balance (position not real nodes)
    data = magnitude_f(local_balance_rbc, positions_of_elements_not_in_boundary, flownetwork.zeroFlowThresholdMagnitude, indices_over_blue)
    values_array = []
    # WRITING PART
    with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'a') as file:
        file.write(f"\n\n------------------------ Iteration {flownetwork.iteration} ------------------------\n\n "
                   f"--- Not crossing nodes of 2MagnitudeThreshold: {len(indices_over_blue)} ({len(indices_over_blue) / len(local_balance_rbc) * 100:.5e}%)"
                   f"\n------------------------\n")
        for magnitudo, numeri_in_magnitudo in data.items():
            if magnitudo < (flownetwork.zeroFlowThresholdMagnitude - 2) and magnitudo != 0:
                file.write(f"Magnitudo {magnitudo}: {numeri_in_magnitudo['count']} values. NODES: {numeri_in_magnitudo['nodes']}\n")
        file.write(f"------------------------------------------------------------\n")

        for index in indices_over_blue:
            output_line = f"node {index} connected with vessels :  {flownetwork.edge_connected_position[index]}\n"
            file.write(output_line)
            value = flownetwork.edge_connected_position[index]
            values_array.append(value)

        # check the indices that does not converge have a change in the flow
        # reflected in a different families dict
        file.write(f"------------------------------------------------------------\n")
        file.write(f"Flow Direction Change [all nodes]:\n")
        for node in positions_of_elements_not_in_boundary:
            # all that have a flow change
            if families_dict_total[node] != flownetwork.families_dict_total[node]:
                vessel = list(set(families_dict_total[node]['par']) ^ set(flownetwork.families_dict_total[node]['par']))[0]
                # file.write(f"New {families_dict_total[node]} - old {flownetwork.families_dict_total[node]}")
                file.write(f"Vessel: {vessel} \n"
                           f"node {node} {families_dict_total[node]} - Connected with vessel par: ")
                for element in families_dict_total[node]["par"]:
                    file.write(f"{flownetwork.flow_rate[element]} ")

                file.write(f' dgs: ')
                for element in families_dict_total[node]["dgs"]:
                    file.write(f"{flownetwork.flow_rate[element]} ")

                file.write(f"\n")
                index = np.where(positions_of_elements_not_in_boundary == node)[0][0]

                file.write(f"Relative Residual: {node_relative_residual[node]}\n"
                           f"Residual : {local_balance_rbc[index]}\n"
                           f"p:[{flownetwork.pressure[node]}\n\n")

                # only the one over the threshold
                if node in indices_over_blue and node not in node_flow_change:
                    node_flow_change.append(node)
                    if vessel not in vessel_flow_change:
                        vessel_flow_change.append(vessel)

                # save all the node with flow changes and vessels
                node_flow_change_total.append(node)
                vessel_flow_change_total.append(vessel)

        flownetwork.indices_over_blue, flownetwork.families_dict_total, flownetwork.node_flow_change, flownetwork.vesel_flow_change, \
            flownetwork.node_flow_change_total, flownetwork.vessel_flow_change_total = \
            indices_over_blue, families_dict_total, np.array(node_flow_change), vessel_flow_change, np.array(node_flow_change_total), vessel_flow_change_total
        file.write(f"\nThe node that have the change of flow and are over the threshold are {node_flow_change_total}\n")
        file.write(f"\nThe vessel that have the change of flow and are over the threshold are {vessel_flow_change_total}\n")
        file.write(f"------------------------------------------------------------\n\n")

        if len(node_flow_change_total) != 0:
            file.write(f"Flow Direction Change [not-converging nodes]\n"
                       f"NODES:{node_flow_change}\nVESSEL:{vessel_flow_change}\n\n"
                       f"Over the threshold are {len(node_flow_change) / len(node_flow_change_total)}/{len(node_flow_change)} nodes and  "
                       f"{len(vessel_flow_change) / len(vessel_flow_change_total)}/{len(vessel_flow_change)} vessels\n")
            for vessel in vessel_flow_change:
                for node in positions_of_elements_not_in_boundary:
                    if vessel in families_dict_total[node]["par"] or vessel in families_dict_total[node]["dgs"]:
                        file.write(f"-------------------------\n")
                        file.write(f'{node}:{families_dict_total[node]} p:[{flownetwork.pressure[node]}\n')
                        if node not in node_flow_change:
                            # print flow in par and daughters
                            file.write(f'Flow rate par: ')
                            if len(families_dict_total[node]["par"]) != 0:
                                for element in families_dict_total[node]["par"]:
                                    file.write(f"{flownetwork.flow_rate[element]}")
                            else:
                                file.write(f'No par')

                            if len(families_dict_total[node]["dgs"]) != 0:
                                file.write(f' dgs: ')
                                for element in families_dict_total[node]["dgs"]:
                                    file.write(f"{flownetwork.flow_rate[element]}")
                            else:
                                file.write(f' no dgs')

                            file.write(f"\n")
                            # print relative residual at that node
                            file.write(f'Relative Residual {node_relative_residual[node]}\n'
                                       f'Residual at node {node}: {node_residual[node]}\n')

                            file.write(f"-------------------------\n")

                            # print residual at that node

        file.write(f"\n------------------------------------------------------------\n")


def magnitude_f(arr, pos, zeroFlowThresholdMagnitude, indices_over):
    magnitudes = {}
    c = 0
    for idx, number in enumerate(arr):
        if pos[idx] in indices_over:
            # Convert the number to scientific notation
            scientific_notation = "{:e}".format(number)
            # Extract the magnitude based on the exponent
            magnitude = abs(int(scientific_notation.split('e')[1]))
            # Create the key in the desired format
            if magnitude == 0 or magnitude > zeroFlowThresholdMagnitude:
                pass
            elif magnitude in magnitudes:
                magnitudes[magnitude]["nodes"].append(pos[idx])
                magnitudes[magnitude]["count"] += 1
                c += 1
            else:
                magnitudes[magnitude] = {"nodes": [pos[idx]], "count": 1}
                c += 1
    return magnitudes


def magnitude_flow(arr):
    magnitudes = {}

    for idx, number in enumerate(arr):
        # Convert the number to scientific notation
        scientific_notation = "{:e}".format(number)
        # Extract the magnitude based on the exponent
        magnitude = abs(int(scientific_notation.split('e')[1]))
        # Create the key in the desired format
        if magnitude in magnitudes:
            magnitudes[magnitude]["nodes"].append(idx)
            magnitudes[magnitude]["count"] += 1
        else:
            magnitudes[magnitude] = {"nodes": [idx], "count": 1}
    return magnitudes


def dict_for_families(flownetwork):
    node_connected = flownetwork.node_connected
    edge_connected_position = flownetwork.edge_connected_position
    pressure = copy.deepcopy(flownetwork.pressure)
    pressure_node = copy.deepcopy(flownetwork.pressure_node)
    node_identifiers = flownetwork.node_identifiers
    families_dict = {node: {"par": [], "dgs": []} for node in node_identifiers}
    vessel_general = []

    for node in pressure_node:
        if node[1] in node_identifiers:
            node_id = node[1]
            for con in range(0, len(node_connected[node_id])):
                edge_position = edge_connected_position[node_id.astype(int)][con]
                if node_id == 4881 or node_id == 4892:
                    print()
                if pressure[node_connected[node_id][con]] == node[0]:
                    print(node_id)
                elif pressure[node_connected[node_id][con]] > node[0]:
                    families_dict[node_id]["par"].append(edge_position)
                    vessel_general.append(edge_position)
                else:
                    families_dict[node_id]["dgs"].append(edge_position)
                    vessel_general.append(edge_position)

    vessel_general.sort()
    return families_dict, vessel_general


def dict_for_families_total(flownetwork):
    node_connected = flownetwork.node_connected
    edge_connected_position = flownetwork.edge_connected_position
    pressure = copy.deepcopy(flownetwork.pressure)
    pressure_node = np.zeros((flownetwork.nr_of_vs, 2))
    # [pressure][node] in a single array
    for pres in range(0, flownetwork.nr_of_vs):
        pressure_node[pres] = np.array([pressure[pres], pres])

    # ordered in base of pressure [pressure][node]
    pressure_node = pressure_node[np.argsort(pressure_node[:, 0])[::-1]]

    families_dict = {node: {"par": [], "dgs": []} for node in range(0, flownetwork.nr_of_vs)}

    for node in pressure_node:
        node_id = node[1].astype(int)
        for nodeCon, edge in zip(node_connected[node_id], edge_connected_position[node_id]):
            # inside I have the edges
            # if pressure[nodeCon] == node[0]:
            #     print(node_id)
            # el
            if pressure[nodeCon] > node[0]:
                families_dict[node_id]["par"].append(edge)
            else:
                families_dict[node_id]["dgs"].append(edge)

    return families_dict
