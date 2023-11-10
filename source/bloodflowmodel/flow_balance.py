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
        flownetwork.positions_of_elements_not_in_boundary = np.where(is_inside_node)[0]
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
            flownetwork.node_relative_residual = {node: [] for node in flownetwork.positions_of_elements_not_in_boundary}
            flownetwork.two_MagnitudeThreshold = 1 * 10 ** (3 - flownetwork.zeroFlowThresholdMagnitude)

        if flownetwork.stop:
            knoledge(self, flownetwork, flownetwork.n_stop, local_balance_rbc)

        elif flownetwork.zeroFlowThreshold is not None and maxBalance <= flownetwork.two_MagnitudeThreshold and flownetwork.boundary_hematocrit[0] == \
                flownetwork.goal:
            flownetwork.stop = True
            knoledge(self, flownetwork, flownetwork.n_stop, local_balance_rbc)

        elif iteration > 1:
            flownetwork.families_dict_total = copy.deepcopy(dict_for_families_total(flownetwork))

        # if iteration % 4000 == 0 and iteration != 0 and flownetwork.increment != 1:
        #     pass

        flownetwork.local_balance_rbc = local_balance_rbc


def knoledge(self, flownetwork, local_balance_rbc, positions_of_elements_not_in_boundary):
    # variable needed
    vessel_general = copy.deepcopy(flownetwork.vessel_general)
    node_relative_residual = copy.deepcopy(flownetwork.node_relative_residual)
    node_residual = copy.deepcopy(flownetwork.node_relative_residual)
    node_flow_change, vessel_flow_change, node_flow_change_total, vessel_flow_change_total, node_relative_residual_plot, node_residual_plot = [], [], [], [], \
        [], []
    # errori negli hd e nei flow

    vessel_value_hd = copy.deepcopy(flownetwork.vessel_value_hd)
    vessel_value_flow = copy.deepcopy(flownetwork.vessel_value_flow)
    flow_rate = abs(copy.deepcopy(flownetwork.flow_rate))

    families_dict_total = copy.deepcopy(dict_for_families_total(flownetwork))

    # salvo i valori di hd e flow per ogni vessel coinvolto
    for vessel in vessel_general:
        vessel_value_hd[vessel].append(flownetwork.hd[vessel])
        vessel_value_flow[vessel].append(flow_rate[vessel])
    flownetwork.vessel_value_hd, flownetwork.vessel_value_flow = vessel_value_hd, vessel_value_flow

    # Comput eht relative residual
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

        index = np.where(positions_of_elements_not_in_boundary == node)
        relative_residual = local_balance_rbc[index][0] / np.average(flow_par + flow_dgs)
        if not np.isnan(relative_residual):
            node_relative_residual_plot.append(relative_residual)
            node_residual_plot.append(local_balance_rbc[index][0])

        node_relative_residual[node].append(relative_residual)
        node_residual[node] = local_balance_rbc[index]
    flownetwork.node_relative_residual, flownetwork.node_residual, flownetwork.positions_of_elements_not_in_boundary, flownetwork.node_relative_residual_plot, flownetwork.node_residual_plot \
        = node_relative_residual, node_residual, positions_of_elements_not_in_boundary, np.array(node_relative_residual_plot), np.array(node_residual_plot)

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
        for index in positions_of_elements_not_in_boundary:
            # all that have a flow change
            if families_dict_total[index] != flownetwork.families_dict_total[index]:
                flow_par, vessel = 0, list(set(families_dict_total[index]['par']) ^ set(flownetwork.families_dict_total[index]['par']))[0]
                # file.write(f"New {families_dict_total[index]} - old {flownetwork.families_dict_total[index]}")
                file.write(f"Vessel: {vessel} \n")
                file.write(f"node {index} {families_dict_total[index]} - Connected with vessel par: ")
                for element in families_dict_total[index]["par"]:
                    file.write(f"{flownetwork.flow_rate[element]} ")
                    flow_par += abs(flow_rate[element])

                file.write(f' dgs: ')
                for element in families_dict_total[index]["dgs"]:
                    file.write(f"{flownetwork.flow_rate[element]} ")
                    flow_par += abs(flow_rate[element])
                file.write(f"\n")
                file.write(f"Relative Residual: {node_relative_residual[index]}\n")

                # only the one over the threshold
                if index in indices_over_blue:
                    node_flow_change.append(index)
                    vessel_flow_change.append(vessel)

                # save all the node with flow changes and vessels
                node_flow_change_total.append(index)
                vessel_flow_change_total.append(vessel)

        flownetwork.indices_over_blue, flownetwork.families_dict_total, flownetwork.node_flow_change, flownetwork.vesel_flow_change, \
            flownetwork.node_flow_change_total, flownetwork.vesel_flow_change_total = \
            indices_over_blue, families_dict_total, np.array(node_flow_change), vessel_flow_change, np.array(node_flow_change_total), vessel_flow_change_total
        file.write(f"\nThe node that have the change of flow and are over the threshold are {node_flow_change_total}\n")
        file.write(f"------------------------------------------------------------\n")

        if len(node_flow_change_total) != 0:
            file.write(f"Flow Direction Change [not-converging nodes]\n"
                       f"NODES:{node_flow_change}\nVESSEL:{vessel_flow_change}\n"
                       f"Over the threshold are {len(node_flow_change) / len(node_flow_change_total)}/{len(node_flow_change)} nodes and  "
                       f"{len(vessel_flow_change) / len(vessel_flow_change_total)}/{len(vessel_flow_change)} vessels\n")
            for vessel in vessel_flow_change:
                for node in positions_of_elements_not_in_boundary:
                    if vessel in families_dict_total[node]["par"] or vessel in families_dict_total[node]["dgs"]:
                        file.write(f"-------------------------\n")
                        file.write(f'{node}:{families_dict_total[node]}\n')
                        if node not in node_flow_change:
                            # print flow in par and daughters
                            file.write(f'Flow rate par: ')
                            if len(families_dict_total[node]["par"]) != 0:
                                for element in families_dict_total[node]["par"]:
                                    file.write(f"{flownetwork.flow_rate[element]} ")
                            else:
                                file.write(f'No par')

                            if len(families_dict_total[node]["dgs"]) != 0:
                                file.write(f' dgs: ')
                                for element in families_dict_total[node]["dgs"]:
                                    file.write(f"{flownetwork.flow_rate[element]} ")
                            else:
                                file.write(f' no dgs')

                            file.write(f"\n")
                            # print relative residual at that node
                            index = np.where(positions_of_elements_not_in_boundary == node)[0][0]
                            file.write(f'Relative Residual {node_relative_residual[node]}\n'
                                       f'Residual at node {node}: {local_balance_rbc[index]}\n')
                            file.write(f"-------------------------\n")

                            # print residual at that node

        file.write(f"\n------------------------------------------------------------\n")
        # print histograms?

        # def knoledge_original(self, flownetwork, n_stop, local_balance_rbc, maxBalance, positions_of_elements_not_in_boundary):
        #     # varibale needed
        #     threshold = flownetwork.zeroFlowThreshold
        #     position_count = np.zeros(flownetwork.nr_of_vs, dtype=int)
        #     families_dict = copy.deepcopy(flownetwork.families_dict)
        #     vessel_general = copy.deepcopy(flownetwork.vessel_general)
        #     node_values = copy.deepcopy(flownetwork.node_values)
        #     node_relative_residual = copy.deepcopy(flownetwork.node_relative_residual)
        #     # errori negli hd e nei flow
        #     node_values_hd = copy.deepcopy(flownetwork.node_values_hd)
        #     node_values_flow = copy.deepcopy(flownetwork.node_values_flow)
        #     vessel_value_hd = copy.deepcopy(flownetwork.vessel_value_hd)
        #     vessel_value_flow = copy.deepcopy(flownetwork.vessel_value_flow)
        #     flow_rate = abs(copy.deepcopy(flownetwork.flow_rate))
        #     output_file_path = f"{self._PARAMETERS['path_output_file']}/{self._PARAMETERS['network_name']}_values.txt"
        #
        #     # salvo i valori di hd e flow per ogni vessel coinvolto
        #     for vessel in vessel_general:
        #         vessel_value_hd[vessel].append(flownetwork.hd[vessel])
        #         vessel_value_flow[vessel].append(flow_rate[vessel])
        #     flownetwork.vessel_value_hd, flownetwork.vessel_value_flow = vessel_value_hd, vessel_value_flow
        #
        #     # salvo i valori di errore
        #     for node in flownetwork.node_identifiers:
        #         hd_par, flow_par, hd_dgs, flow_dgs = 0, 0, 0, 0
        #         # sommo padri e figli di uno stesso nodo
        #         for element in families_dict[node]["par"]:
        #             hd_par += flownetwork.hd[element]
        #             flow_par += flow_rate[element]
        #
        #         for dato in families_dict[node]["dgs"]:
        #             hd_dgs += flownetwork.hd[dato]
        #             flow_dgs += flow_rate[dato]
        #
        #         node_values_hd[node].append(abs(hd_par - hd_dgs))
        #         node_values_flow[node].append(abs(flow_par - flow_dgs))
        #         index = np.where(positions_of_elements_not_in_boundary == node)
        #         relative_residual = local_balance_rbc[index] / np.average(flow_par + flow_dgs)
        #         node_relative_residual[node].append(relative_residual)
        #     flownetwork.node_values_hd, flownetwork.node_values_flow, flownetwork.node_relative_residual = node_values_hd, node_values_flow, node_relative_residual
        #
        #     local_balance_rbc_corr = np.zeros(flownetwork.nr_of_vs)
        #     for node in range(0, flownetwork.nr_of_vs):
        #         index = np.where(positions_of_elements_not_in_boundary == node)
        #         local_balance_rbc_corr[index] = local_balance_rbc[index]
        #
        #     if n_stop == 1:  # TODO: if we want the first iterations 0
        #         with open(output_file_path, 'a') as file:
        #             file.write(f"First crossing of threshold from MeanBalance at iteration {flownetwork.iteration} \n")
        #             # f"MaxBalance has number of not crossing vessel: {count_over_threshold} ({percentage_over_threshold:.5e}%)\n")
        #
        #     # sono sotto il threshold
        #     elif maxBalance < threshold:
        #         with open(output_file_path, 'a') as file:
        #             file.write(f"Crossing at {flownetwork.iteration} it.\n")
        #
        #     # ho delle cose sotto anche se ho oltrepassato
        #     else:
        #         # with open(output_file_path, 'a') as file:
        #         #     file.write(f"Not crossing - {flownetwork.iteration} it.:\n"
        #         #                f"MaxBalance has number of not crossing vessel: {count_over_threshold} ({percentage_over_threshold:.5e}%)\n")
        #         #     for i, value in enumerate(local_balance_rbc):
        #         #         if value >= threshold:
        #         #             file.write(f"Node {i} with residual {value:.5e}\n")
        #
        #         # salvataggio di quelli che appaiono di più
        #         # indici di quelli che sono sopra il threshold
        #         # indices_over = np.argpartition(local_balance_rbc, -len(local_balance_rbc[local_balance_rbc > threshold]))[-len(local_balance_rbc[local_balance_rbc
        #         # >= threshold]):]
        #         # new array to store the nodes (real) that are over the threshold of residuals
        #         # the number is known based on previous computation
        #         # # count over the threshold for all the vessels
        #         # count_over_threshold = np.sum(local_balance_rbc > threshold)
        #         # percentage_over_threshold = (count_over_threshold / len(local_balance_rbc)) * 100
        #
        #         # count the one over
        #         blue_Threshold = 1 * 10 ** (3 - flownetwork.zeroFlowThresholdMagnitude)
        #
        #         #
        #         indices_over, c = [], 0
        #         indices_over_blue, d = [], 0
        #
        #         for node in positions_of_elements_not_in_boundary:
        #             index = np.where(positions_of_elements_not_in_boundary == node)
        #             if local_balance_rbc[index] > threshold and local_balance_rbc[index] != 0:
        #                 indices_over.append(node)
        #             if local_balance_rbc[index] > blue_Threshold and local_balance_rbc[index] != 0:
        #                 indices_over_blue.append(node)
        #
        #         indices_over, indices_over_blue = np.array(indices_over), np.array(indices_over_blue)
        #
        #         # Count the frequency of each position
        #         for position in range(flownetwork.nr_of_vs):
        #             if position in indices_over:
        #                 position_count[position] += 1
        #
        #         # guardo i residui
        #         for node in flownetwork.node_identifiers:
        #             node_values[node] = np.append(node_values[node], local_balance_rbc[node])
        #         flownetwork.node_values = node_values
        #
        #         flownetwork.position_count = position_count
        #
        #         if flownetwork.n_stop == 5:
        #             # I have the pure magnitudes for each values of the local balance (position not real nodes)
        #             data = magnitude_f(local_balance_rbc, positions_of_elements_not_in_boundary, flownetwork.zeroFlowThresholdMagnitude, indices_over)
        #             values_array = []
        #
        #             #
        #             unique_values = set()
        #             for index in indices_over:
        #                 value = set(flownetwork.edge_connected_position[index])
        #                 unique_values = (unique_values.union(value))
        #
        #             abs_flow = flow_rate[list(unique_values)]
        #
        #             # WRITING PART
        #             with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'a') as file:
        #                 file.write(f"Interval of RESIDUAL\n"
        #                            f"Max: {max(local_balance_rbc)} Min: {min(local_balance_rbc[local_balance_rbc > threshold])}\n")
        #
        #                 file.write(
        #                     f"------------------------ Not crossing nodes of zeroFlowThreshold: {len(indices_over)} ("
        #                     f"{len(indices_over) / len(local_balance_rbc) * 100 :.5e}%)"
        #                     f"------------------------\n")
        #                 # riscrivere incongruenza
        #                 for magnitudo, numeri_in_magnitudo in data.items():
        #                     file.write(f"Magnitudo {magnitudo}: {numeri_in_magnitudo['count']} values. NODES: {numeri_in_magnitudo['nodes']}\n")
        #                 file.write(f"------------------------------------------------------------\n")
        #
        #                 for index in indices_over:
        #                     output_line = f"node {index} connected with vessels :  {flownetwork.edge_connected_position[index]}\n"
        #                     file.write(output_line)
        #                     value = flownetwork.edge_connected_position[index]
        #                     values_array.append(value)
        #                 file.write(f"------------------------------------------------------------\n")
        #
        #                 file.write(
        #                     f"------------------------ Not crossing nodes of 2MagnitudeThreshold: {len(indices_over_blue)} ({len(indices_over_blue) / len(local_balance_rbc) * 100:.5e}%)"
        #                     f"\n------------------------\n")
        #                 for magnitudo, numeri_in_magnitudo in data.items():
        #                     if magnitudo < (flownetwork.zeroFlowThresholdMagnitude - 2) and magnitudo != 0:
        #                         file.write(f"Magnitudo {magnitudo}: {numeri_in_magnitudo['count']} values. NODES: {numeri_in_magnitudo['nodes']}\n")
        #                 file.write(f"------------------------------------------------------------\n")
        #
        #                 for index in indices_over_blue:
        #                     output_line = f"node {index} connected with vessels :  {flownetwork.edge_connected_position[index]}\n"
        #                     file.write(output_line)
        #                     value = flownetwork.edge_connected_position[index]
        #                     values_array.append(value)
        #
        #                 file.write(f"Edge connected to the non converging nodes: {unique_values} \n")
        #                 file.write(f"------------------------------------------------------------\n")
        #
        #                 file.write(f"Values of the flow of those vessel \n")
        #                 file.write(f"FLOW: Max {max(abs_flow)} min:{min(abs_flow[abs_flow != 0])} in abs\n")
        #
        #                 # Sospendo il flow per ora
        #                 # abs_flow_data = magnitude_flow(flow_rate)
        #                 # for magnitudo, numeri_in_magnitudo in abs_flow_data.items():
        #                 #     if numeri_in_magnitudo['nodes'] in
        #                 #         file.write(f"Magnitudo {magnitudo}: {numeri_in_magnitudo['count']} values. VESSEL: {numeri_in_magnitudo['nodes']}\n")
        #                 # file.write(f"------------------------------------------------------------\n")
        #                 # for index in unique_values:
        #                 #     output_line = f"vessel {index} flow_rate :  {flownetwork.flow_rate[index]} \n"
        #                 #     file.write(output_line)
        #                 # file.write(f"------------------------------------------------------------\n")
        #                 flownetwork.indices_over, flownetwork.indices_over_blue, flownetwork.local_balance_rbc_corr = indices_over, indices_over_blue, local_balance_rbc_corr


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
                if pressure[node_connected[node_id][con]] > node[0]:
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
    pressure_node = copy.deepcopy(flownetwork.pressure_node)
    families_dict = {node: {"par": [], "dgs": []} for node in range(0, flownetwork.nr_of_vs)}

    for node in pressure_node:
        node_id = node[1]
        for con in range(0, len(node_connected[node_id])):
            edge_position = edge_connected_position[node_id.astype(int)][con]
            # inside I have the edges
            if pressure[node_connected[node_id][con]] > node[0]:
                families_dict[node_id]["par"].append(edge_position)
            else:
                families_dict[node_id]["dgs"].append(edge_position)

    return families_dict
