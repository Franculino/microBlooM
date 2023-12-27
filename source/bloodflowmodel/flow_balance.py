import copy
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

    @staticmethod
    def _get_flow_balance_rbcs(flownetwork):

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
        """
        Function to check the mass-flow balance of the network
        @param flownetwork: flow network object
        @type flownetwork: source.flow_network.FlowNetwork
        """
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

        # Zero-flow-threshold
        # The zero flow threshold is set as the max of the mass balance error for the internal nodes
        # it is computed the new flow inside
        if self._PARAMETERS["ZeroFlowThreshold"] is True and flownetwork.zeroFlowThreshold is None:
            flownetwork.flow_rate = set_low_flow_threshold(flownetwork, local_balance)

        # Compute the max residual of RBCs
        # RBC balance
        flow_rbcs = self._get_flow_balance_rbcs(flownetwork)
        local_balance_rbc = np.abs(flow_rbcs[is_inside_node])
        maxBalance, meanBalance = max(local_balance_rbc), np.mean(local_balance_rbc)
        flownetwork.local_balance_rbc = local_balance_rbc
        flownetwork.maxBalance = maxBalance

        # ----- Iterative procedure with our convergence criteria -----
        if self._PARAMETERS["iterative_routine"] == 2:
            if iteration > 2:
                flownetwork.node_relative_residual = {node: [] for node in positions_of_elements_not_in_boundary}
                flownetwork.two_MagnitudeThreshold = 1 * 10 ** (3 - flownetwork.zeroFlowThresholdMagnitude)

                # if the iteration number reach 4000 is considered as not convergent
                if iteration == 4000:
                    flownetwork.our_convergence_criteria = True
                    self.knowledge(flownetwork, local_balance_rbc, positions_of_elements_not_in_boundary)

                # Convergence reached
                elif flownetwork.zeroFlowThreshold is not None and iteration > 2 and maxBalance <= flownetwork.two_MagnitudeThreshold:
                    flownetwork.our_convergence_criteria = True

                elif iteration == 1:
                    flownetwork.families_dict_total = copy.deepcopy(dict_for_families_total(flownetwork))

    def knowledge(self, flownetwork, local_balance_rbc, positions_of_elements_not_in_boundary):
        """
        The function is designed to print all the relevant information about a non converging network.
        The network is defined as non converging if it doesn't reach convergence after 4000 iterations.

        @param flownetwork:flow network object
        @type flownetwork: source.flow_network.FlowNetwork
        @param local_balance_rbc: local balance of rbcs
        @type local_balance_rbc: np.array
        @param positions_of_elements_not_in_boundary: internal element of the network
        @type positions_of_elements_not_in_boundary: np.array
        """
        # VARIABLES for the function
        node_relative_residual, node_residual = np.zeros(flownetwork.nr_of_vs), np.zeros(flownetwork.nr_of_vs)
        node_flow_change, vessel_flow_change, node_flow_change_total, vessel_flow_change_total = [], [], [], []
        flow_rate = abs(copy.deepcopy(flownetwork.flow_rate))
        families_dict_total = copy.deepcopy(dict_for_families_total(flownetwork))

        # Compute the node relative residual and store the local_balance for our node
        for node in positions_of_elements_not_in_boundary:
            # Save the flow of parents and daughter for each internal node
            hd_par, flow_par, hd_dgs, flow_dgs = 0, 0, 0, 0

            for element in families_dict_total[node]["par"]:
                hd_par += flownetwork.hd[element]
                flow_par += flow_rate[element]

            for dato in families_dict_total[node]["dgs"]:
                hd_dgs += flownetwork.hd[dato]
                flow_dgs += flow_rate[dato]

            # Positions of elements not in boundary are mapped on local_balance RBCs.
            # The index of the node will be the same one to look into for the local balance
            index = np.where(positions_of_elements_not_in_boundary == node)
            flow_gather = flow_par + flow_dgs
            if flow_gather != 0:
                relative_residual = local_balance_rbc[index][0] / np.average(flow_gather)
                node_relative_residual[node] = relative_residual
            else:
                node_relative_residual[node] = 0
            node_residual[node] = local_balance_rbc[index]

        # Store for future
        flownetwork.node_relative_residual, flownetwork.node_residual, flownetwork.positions_of_elements_not_in_boundary = \
            node_relative_residual, node_residual, positions_of_elements_not_in_boundary

        # Indices over the Threshold
        indices_over_blue = []
        for node in positions_of_elements_not_in_boundary:
            index = np.where(positions_of_elements_not_in_boundary == node)
            if local_balance_rbc[index] > flownetwork.two_MagnitudeThreshold and local_balance_rbc[index] != 0:
                indices_over_blue.append(node)
        indices_over_blue = np.array(indices_over_blue)

        # ----- WRITING SECTION -----
        # I have the understanding the magnitudes for each value of the local balance (position not real nodes)
        data = self.magnitude_f(local_balance_rbc, positions_of_elements_not_in_boundary, flownetwork.zeroFlowThresholdMagnitude, indices_over_blue)
        values_array = []
        with open(self._PARAMETERS['path_output_file'] + "/" + self._PARAMETERS['network_name'] + ".txt", 'a') as file:

            # write at current iteration the nodes that doesn't converge and the percentage of them
            file.write(f"\n\n------------------------ Iteration {flownetwork.iteration} ------------------------\n\n "
                       f"--- Not crossing nodes of 2MagnitudeThreshold: {len(indices_over_blue)} ({len(indices_over_blue) / len(local_balance_rbc) * 100:.5e}%)"
                       f"\n------------------------\n")

            # the writing is also reported for each magnitude
            for magnitudo, numeri_in_magnitudo in data.items():
                if magnitudo < flownetwork.two_MagnitudeThreshold and magnitudo != 0:
                    file.write(f"Magnitudo {magnitudo}: {numeri_in_magnitudo['count']} values. NODES: {numeri_in_magnitudo['nodes']}\n")
            file.write(f"------------------------------------------------------------\n")

            # the nodes are also reported with the vessels that are connected to
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
                indices_over_blue, families_dict_total, np.array(node_flow_change), vessel_flow_change, np.array(
                    node_flow_change_total), vessel_flow_change_total
            file.write(f"\nThe node that have the change of flow and are over the threshold are {node_flow_change_total}\n")
            file.write(f"\nThe vessel that have the change of flow and are over the threshold are {vessel_flow_change_total}\n")
            file.write(f"------------------------------------------------------------\n\n")

            # Analysis on different flow directions
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
            file.write(f"\n------------------------------------------------------------\n")

    @staticmethod
    def magnitude_f(arr, pos, zeroFlowThresholdMagnitude, indices_over):
        """
        Function to return the magnitude of an array and return an array
        with the node correspond to that magnitude and the number of elements for that.
        """
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


def dict_for_families_total(flownetwork):
    """
    Function to return the connection for each node
    par: parent nodes
    dgs: daughter nodes
    """
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
            if pressure[nodeCon] > node[0]:
                families_dict[node_id]["par"].append(edge)
            else:
                families_dict[node_id]["dgs"].append(edge)

    return families_dict
