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

        flow_balance, flow_rbcs = np.zeros(nr_of_vs), np.zeros(nr_of_vs)
        flow_rate = copy.deepcopy(flownetwork.flow_rate)
        hd = copy.deepcopy(flownetwork.hd)

        for eid in range(nr_of_es):
            flow_balance[edge_list[eid, 0]] += flow_rate[eid]
            flow_balance[edge_list[eid, 1]] -= flow_rate[eid]

        for eid in range(nr_of_es):
            flow_rbcs[edge_list[eid, 0]] += np.multiply(flow_rate[eid], hd[eid])
            flow_rbcs[edge_list[eid, 1]] -= np.multiply(flow_rate[eid], hd[eid])

        return flow_balance, flow_rbcs

    def check_flow_balance(self, flownetwork, tol=1.00E-5):

        nr_of_vs = flownetwork.nr_of_vs
        flow_rate = flownetwork.flow_rate
        boundary_vs = flownetwork.boundary_vs

        flow_balance, flow_rbcs = self._get_flow_balance(flownetwork)

        ref_flow = np.abs(flow_rate[boundary_vs[0]])
        tol_flow = tol * ref_flow

        is_inside_node = np.logical_not(np.in1d(np.arange(nr_of_vs), boundary_vs))
        local_balance = np.abs(flow_balance[is_inside_node])
        is_locally_balanced = local_balance < tol_flow

        # RBC
        local_balance_rbc = np.abs(flow_rbcs[is_inside_node])
        maxBalance, meanBalance = max(local_balance_rbc), np.mean(local_balance_rbc)
        flownetwork.residualOverIterationMax = np.append(flownetwork.residualOverIterationMax, maxBalance)
        flownetwork.residualOverIterationNorm = np.append(flownetwork.residualOverIterationNorm, meanBalance)

        if flownetwork.stop:
            # sono già passato e conto per vedere cosa succede nelle prossime iterazioni
            flownetwork.n_stop += 1
            knoledge(self, flownetwork, flownetwork.n_stop, local_balance_rbc, maxBalance)

        elif (flownetwork.zeroFlowThreshold is not None and meanBalance < flownetwork.zeroFlowThreshold) or flownetwork.iteration > 2300:  # 2300
            # segnalo che sono passato la prima volta
            flownetwork.stop = True
            # parto da uno a contare per vedere cosa accade
            flownetwork.n_stop = 1
            knoledge(self, flownetwork, flownetwork.n_stop, local_balance_rbc, maxBalance)

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

        if flownetwork.families_dict is None and flownetwork.iteration > 2:
            flownetwork.families_dict, flownetwork.vessel_general = copy.deepcopy(dict_for_families(flownetwork))
            flownetwork.node_values = {node: [] for node in flownetwork.node_identifiers}
            flownetwork.node_values_hd = {node: [] for node in flownetwork.node_identifiers}
            flownetwork.node_values_flow = {node: [] for node in flownetwork.node_identifiers}
            flownetwork.vessel_value_hd = {vessel: [] for vessel in flownetwork.vessel_general}
            flownetwork.vessel_value_flow = {vessel: [] for vessel in flownetwork.vessel_general}


def knoledge(self, flownetwork, n_stop, local_balance_rbc, maxBalance):
    # varibale needed
    threshold = copy.deepcopy(flownetwork.zeroFlowThreshold)
    position_count = np.zeros(flownetwork.nr_of_vs, dtype=int)
    families_dict = copy.deepcopy(flownetwork.families_dict)
    vessel_general = copy.deepcopy(flownetwork.vessel_general)
    node_values = copy.deepcopy(flownetwork.node_values)
    # errori negli hd e nei flow
    node_values_hd = copy.deepcopy(flownetwork.node_values_hd)
    node_values_flow = copy.deepcopy(flownetwork.node_values_flow)
    vessel_hd = copy.deepcopy(flownetwork.vessel_value_hd)
    vessel_flow = copy.deepcopy(flownetwork.vessel_value_flow)
    flow_rate = abs(copy.deepcopy(flownetwork.flow_rate))
    output_file_path = f"{self._PARAMETERS['path_output_file']}/{self._PARAMETERS['network_name']}_values.txt"

    # salvo i valori di hd e flow per ogni vessel coinvolto
    for vessel in vessel_general:
        vessel_hd[vessel].append(flownetwork.hd[vessel])
        vessel_flow[vessel].append(flow_rate[vessel])
    flownetwork.vessel_value_hd, flownetwork.vessel_value_flow = vessel_hd, vessel_flow

    # salvo i valori di errore
    for node in flownetwork.node_identifiers:
        hd_par, flow_par, hd_dgs, flow_dgs = 0, 0, 0, 0
        # sommo padri e figli di uno stesso nodo
        for element in families_dict[node]["par"]:
            hd_par += flownetwork.hd[element]
            flow_par += flow_rate[element]

        for dato in families_dict[node]["dgs"]:
            hd_dgs += flownetwork.hd[dato]
            flow_dgs += flow_rate[dato]

        node_values_hd[node].append(abs(hd_par - hd_dgs))
        node_values_flow[node].append(abs(flow_par - flow_dgs))
    flownetwork.node_values_hd, flownetwork.node_values_flow = node_values_hd, node_values_flow

    # node_values = copy.deepcopy(flownetwork.node_values)
    # node_identifiers = [298, 312, 316, 319, 354, 367, 370, 371, 374, 395]
    # node_identifiers = [2801, 5161, 6639, 1759, 93]
    # node_identifiers = [1865, 2429, 2, 3363, 5382]
    # node_identifiers = [595, 11469, 701, 2289, 371]
    # node_identifiers = [384, 11914, 3457, 6671, 9440]
    count_over_threshold = np.sum(local_balance_rbc > threshold)
    percentage_over_threshold = (count_over_threshold / len(local_balance_rbc)) * 100

    if n_stop == 1:
        with open(output_file_path, 'a') as file:
            file.write(f"First crossing of threshold from MeanBalance at iteration {flownetwork.iteration} \n"
                       f"MaxBalance has number of not crossing vessel: {count_over_threshold} ({percentage_over_threshold:.5e}%)\n")
    # sono sotto il threshold
    elif maxBalance < threshold:
        with open(output_file_path, 'a') as file:
            file.write(f"Crossing at {flownetwork.iteration} it.\n")
    # ho delle cose sotto anche se ho oltrepassato
    else:
        with open(output_file_path, 'a') as file:
            file.write(f"Not crossing - {flownetwork.iteration} it.:\n"
                       f"MaxBalance has number of not crossing vessel: {count_over_threshold} ({percentage_over_threshold:.5e}%)\n")
            for i, value in enumerate(local_balance_rbc):
                if value >= threshold:
                    file.write(f"Node {i} with residual {value:.5e}\n")

        # salvataggio di quelli che appaiono di più
        # indici di quelli che sono sopra il threshold
        indices_over = np.argpartition(local_balance_rbc, -len(local_balance_rbc[local_balance_rbc > threshold]))[
                       -len(local_balance_rbc[local_balance_rbc >= threshold]):]

        # Count the frequency of each position
        for position in set(range(flownetwork.nr_of_vs)):
            count_arr = (position in indices_over)
            position_count[position] += count_arr
        # Ensure that all positions have a count of + 0
        for position in range(max(set(range(flownetwork.nr_of_vs))) + 1):
            if position not in position_count:
                position_count[position] += 0

        with open(output_file_path, 'a') as file:
            for index in indices_over:
                count = position_count[index]
                if count != 0:
                    output_line = f"{index}: {count} - "
                    file.write(output_line)
            file.write("\n")

        # guardo i residui
        # for node in node_identifiers:
        #     node_values[node] = np.append(node_values[node], local_balance_rbc[node])
        # flownetwork.node_values = node_values

        flownetwork.position_count = position_count

        if flownetwork.n_stop == 100:
            data = magnitudo_f(local_balance_rbc[local_balance_rbc > threshold])
            values_array = []

            unique_values = set()
            for index in indices_over:
                value = set(flownetwork.edge_connected_position[index])
                unique_values = (unique_values.union(value))

            abs_flow = flow_rate[list(unique_values)]

            # WRITING PART
            with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'a') as file:
                file.write(f"Interval of RESIDUAL\n"
                           f"not crossing vessel: {count_over_threshold} ({percentage_over_threshold:.5e}%)\n"
                           f"Max: {max(local_balance_rbc)} Min: {min(local_balance_rbc[local_balance_rbc > threshold])}\n")

                for magnitudo, numeri_in_magnitudo in data.items():
                    file.write(f"Magnitudo {magnitudo}: {numeri_in_magnitudo['count']} values. NODES: {numeri_in_magnitudo['positions']}\n")
                file.write(f"------------------------------------------------------------\n")

                for index in indices_over:
                    output_line = f"node {index} connected with vessels :  {flownetwork.edge_connected_position[index]} \n"
                    file.write(output_line)
                    value = flownetwork.edge_connected_position[index]
                    values_array.append(value)
                file.write(f"------------------------------------------------------------\n")

                file.write(f"Edge connected to the non converging nodes: {unique_values} \n")
                file.write(f"------------------------------------------------------------\n")

                file.write(f" Values of the flow of those vessel \n")
                file.write(f"FLOW: Max {max(abs_flow)} min:{min(abs_flow[abs_flow != 0])} in abs\n")
                abs_flow_data = magnitudo_f(abs_flow)
                for magnitudo, numeri_in_magnitudo in abs_flow_data.items():
                    file.write(f"Magnitudo {magnitudo}: {numeri_in_magnitudo['count']} values. VESSEL: {numeri_in_magnitudo['positions']}\n")
                file.write(f"------------------------------------------------------------\n")
                for index in unique_values:
                    output_line = f" vessel {index} flow_rate :  {flownetwork.flow_rate[index]} \n"
                    file.write(output_line)
                file.write(f"------------------------------------------------------------\n")


def magnitudo_f(arr):
    magnitudes = {}

    for idx, number in enumerate(arr):
        # Convert the number to scientific notation
        scientific_notation = "{:e}".format(number)
        # Extract the magnitude based on the exponent
        magnitude = abs(int(scientific_notation.split('e')[1]))
        # Create the key in the desired format
        formatted_key = f'1e-{magnitude}'
        if formatted_key in magnitudes:
            magnitudes[formatted_key]["positions"].append(idx)
            magnitudes[formatted_key]["count"] += 1
        else:
            magnitudes[formatted_key] = {"positions": [idx], "count": 1}
    return magnitudes


def dict_for_families(flownetwork):
    node_connected = copy.deepcopy(flownetwork.node_connected)
    edge_connected_position = copy.deepcopy(flownetwork.edge_connected_position)
    pressure = copy.deepcopy(flownetwork.pressure)
    pressure_node = copy.deepcopy(flownetwork.pressure_node)
    families_dict = {node: {"par": [], "dgs": []} for node in flownetwork.node_identifiers}
    vessel_general = []

    for node in pressure_node:

        if node[1] in flownetwork.node_identifiers:

            for con in range(0, len(node_connected[node[1]])):

                if pressure[node_connected[node[1]][con]] >= node[0]:

                    families_dict[node[1]]["par"].append(edge_connected_position[node[1].astype(int)][con])
                    vessel_general.append(edge_connected_position[node[1].astype(int)][con])

                else:
                    families_dict[node[1]]["dgs"].append(edge_connected_position[node[1].astype(int)][con])
                    vessel_general.append(edge_connected_position[node[1].astype(int)][con])

    vessel_general.sort()
    return families_dict, vessel_general
