from abc import ABC, abstractmethod
from types import MappingProxyType
import sys

import math
import numpy as np
from math import e
import copy

from matplotlib import pyplot as plt
from scipy.special import expit as logistic

import source.bloodflowmodel.transmissibility as transmissibility
import source.bloodflowmodel.pressure_flow_solver as pressureflowsolver
import source.bloodflowmodel.build_system as buildsystem
import source.bloodflowmodel.rbc_velocity as rbc_velocity
from util_methods.util_plot import util_display_graph, graph_creation


def _logit(x):
    return math.log(x / (1 - x))


class DischargeHaematocrit(ABC):
    """
    Abstract base class for the implementations related to calculating the discharge haematocrit.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of DischargeHaematocrit
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS
        self.x_o_init = 1.12  # micrometers
        self.A_o_init = 15.47  # micrometers
        self.B_o_init = 8.13  # micrometers

    @abstractmethod
    def update_hd(self, flownetwork):
        """
        Abstract method to update the discharge haematocrit in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def qRCS(self, flownetwork, case, flow_single, flow_a, flow_b, flow_c, hemat_single, hemat_a, hemat_b, hemat_c):
        """
        check the RBC balance
        qRBC = q * Hdt

        Massbalances:
        q_p1 + q_p2 = q_d,
        qRBC_p1 + qRBC_p2 = qRBC_d1,
        """


class DischargeHaematocritNewtonian(DischargeHaematocrit):
    """
    Class for updating the discharge haematocrit without taking red blood cells into account.
    """

    def update_hd(self, flownetwork):
        """
        Update the discharge haematocrit in the flow network with the zero vector for Newtonian flow.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.hd = flownetwork.ht


class DischargeHaematocritVitroPries1992(DischargeHaematocrit):
    """
    Class for updating the discharge haematocrit based on the empirical in vitro functions by
    Pries, Neuhaus, Gaehtgens (1992).
    """

    def update_hd(self, flownetwork):
        """
        Update the discharge haematocrit in flownetwork based on tube haematocrit and vessel diameter. The
        model==based on the empirical in vitro functions by Pries, Neuhaus, Gaehtgens (1992).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        ht = flownetwork.ht  # Tube haematocrit
        diameter_um = 1.e6 * flownetwork.diameter  # Vessel diameter in micro meters

        x_tmp = 1. + 1.7 * np.exp(-0.35 * diameter_um) - 0.6 * np.exp(-0.01 * diameter_um)  # Eq. (9) in paper
        x_bound = np.copy(x_tmp)
        x_bound[x_tmp > 0.99] = 0.99  # Bound x to values < 1. Equation in paper==only valid for x < 1.

        hd = -x_bound / (2 - 2 * x_bound) + np.sqrt(
            np.square(x_bound / (2 - 2 * x_bound)) + ht / (1 - x_bound))  # Eq 10 in paper
        hd[x_tmp > 0.99] = ht[x_tmp > 0.99]  # For very small and very large diameters: set ht=hd

        flownetwork.hd = hd  # Update discharge haematocrit


class DischargeHaematocritVitroPries2005(DischargeHaematocrit):
    """
    Class for updating the discharge haematocrit based on the empirical in vitro functions by
    Pries and Secomb (2005).
    """

    def update_hd(self, flownetwork):
        """
        Update the discharge haematocrit in flownetwork based on tube haematocrit and vessel diameter. The
        model==based on the empirical in vitro functions by Pries and Secomb (2005).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        ht = flownetwork.ht  # Tube haematocrit
        diameter_um = 1.e6 * flownetwork.diameter  # Vessel diameter in micro meters

        x_tmp = 1. + 1.7 * np.exp(-0.415 * diameter_um) - 0.6 * np.exp(-0.011 * diameter_um)  # From Eq.(1) in paper
        x_bound = np.copy(x_tmp)
        x_bound[x_tmp > 0.99] = 0.  # Bound x to values < 1. Equation in paper==only valid for x < 1.

        hd = -x_bound / (2 - 2 * x_bound) + np.sqrt(
            np.square(x_bound / (2 - 2 * x_bound)) + ht / (1 - x_bound))
        hd[x_tmp > 0.99] = ht[x_tmp > 0.99]  # For very small and very large diameters: set ht=hd

        flownetwork.hd = hd  # Update discharge haematocrit


class DischargeHaematocritPries1990(DischargeHaematocrit):

    def qRCS(self, flownetwork, case, flow_single, flow_a, flow_b, flow_c, hemat_single, hemat_a, hemat_b, hemat_c):
        """
        check the RBC balance
        qRBC = q * Hdt

        Massbalances:
        q_p1 + q_p2 = q_d,
        qRBC_p1 + qRBC_p2 = qRBC_d1,
        """

        tollerance = 1.00E-05
        RBCbalance = 1
        match case:

            # one parent and two daughter (-<)
            case 1:
                flow_parent_suppose = flow_a + flow_b
                if np.abs(flow_single - flow_parent_suppose) <= tollerance:  # check same flow
                    qRBC_a = flow_a * hemat_a
                    qRBC_b = flow_b * hemat_b
                    qRBC_p = flow_single * hemat_single
                    qRBC_p_to_check = qRBC_a + qRBC_b
                    ass = np.abs(qRBC_p_to_check - qRBC_p)
                    if ass <= tollerance:  # check same RBCs
                        RBCbalance = 0

            # one parent and one daughter (-ø-)
            case 2:
                val = np.abs(flow_single - flow_a)
                if np.abs(flow_single - flow_a) <= tollerance:  # check same flow
                    qRBC_single = flow_single * hemat_single
                    qRBC_a = flow_a * hemat_a
                    value = np.abs(qRBC_single - qRBC_a)
                    if np.abs(qRBC_single - qRBC_a) <= tollerance:
                        RBCbalance = 0

            # two parent and one daughter (>-)
            case 3:
                flow_daughter_to_check = flow_a + flow_b

                if np.abs(flow_single - flow_daughter_to_check) <= tollerance:  # check same flow
                    qRBC_a = flow_a * hemat_a
                    qRBC_b = flow_b * hemat_b
                    qRBC_d = flow_single * hemat_single
                    qRBC_d_to_check = qRBC_a + qRBC_b
                    if np.abs(qRBC_d - qRBC_d_to_check) <= tollerance:
                        RBCbalance = 0

            # three parents and one daughter (∃ø-)
            case 4:
                flow_daughter_to_check = flow_a + flow_b + flow_c
                if np.abs(flow_single - flow_daughter_to_check) <= tollerance:  # check same flow
                    qRBC_a = flow_a * hemat_a
                    qRBC_b = flow_b * hemat_b
                    qRBC_c = flow_c * hemat_c
                    qRBC_d = flow_single * hemat_single
                    qRBC_d_to_check = qRBC_a + qRBC_b + qRBC_c
                    if np.abs(qRBC_d - qRBC_d_to_check) <= tollerance:
                        RBCbalance = 0
            case 5:
                print()
            case _:  # no daughter
                RBCbalance = 0

        if RBCbalance == 1:
            print("NOT BALANCED")
        return RBCbalance

    def non_dimentional_param(self, hemat_par, diam_par, diam_a, diam_b):
        diam_a, diam_b, diam_par = diam_a * 1E6, diam_b * 1E6, diam_par * 1E6

        x_0 = self.x_o_init * (1 - hemat_par) / diam_par

        A = (-self.A_o_init) * ((pow(diam_a, 2) - pow(diam_b, 2)) / (pow(diam_a, 2) + pow(diam_b, 2))) * (
                1 - hemat_par) / diam_par

        B = 1 + (self.B_o_init * (1 - hemat_par) / diam_par)

        return x_0, A, B

    def get_erythrocyte_fraction(self, hemat_par, diam_par, diam_a, diam_b, flow_a, flow_parent, flow_b,
                                 fractional_a_qRBCs, fractional_b_qRBCs, fractional_a_blood, fractional_b_blood,
                                 hemat_parent_plot):
        """
        to calculate the fraction of erythrocyte that goes in each daugheter vessel
        """
        fractional_flow_a = flow_a / (flow_a + flow_b)
        fractional_flow_b = flow_b / (flow_a + flow_b)

        x_0, A, B = self.non_dimentional_param(hemat_par, diam_par, diam_a, diam_b)

        qRBCp = hemat_par * flow_parent

        if fractional_flow_a <= x_0:
            fractional_qRBCa, fractional_qRBCb = 0, 1

        elif x_0 < fractional_flow_a < (1 - x_0):
            internal_logit = (fractional_flow_a - x_0) / (1 - x_0)
            logit_result = A + B * _logit(internal_logit)
            fractional_qRBCa = (pow(e, logit_result) / (1 + pow(e, logit_result)))
            fractional_qRBCb = 1 - fractional_qRBCa

        elif fractional_flow_a >= (1 - x_0):
            fractional_qRBCa, fractional_qRBCb = 1, 0

        hemat_a = (fractional_qRBCa * qRBCp) / flow_a
        hemat_b = (fractional_qRBCb * qRBCp) / flow_b

        threshold = 0.99
        if hemat_b >= threshold:
            hemat_surplus = hemat_b - threshold
            fractional_RBCs_suprlus = (hemat_surplus * flow_b) / qRBCp

            fractional_qRBCb = fractional_qRBCb - fractional_RBCs_suprlus
            fractional_qRBCa = fractional_qRBCa + fractional_RBCs_suprlus

            hemat_a = (fractional_qRBCa * qRBCp) / flow_a
            hemat_b = (fractional_qRBCb * qRBCp) / flow_b
            # for plot

        elif hemat_a >= threshold:
            hemat_surplus = hemat_a - threshold
            fractional_RBCs_suprlus = (hemat_surplus * flow_a) / qRBCp

            # for plot
            fractional_qRBCb = fractional_qRBCb + fractional_RBCs_suprlus
            fractional_qRBCa = fractional_qRBCa - fractional_RBCs_suprlus
            hemat_a = (fractional_qRBCa * qRBCp) / flow_a
            hemat_b = (fractional_qRBCb * qRBCp) / flow_b

        fractional_a_qRBCs.append(fractional_qRBCa)
        fractional_b_qRBCs.append(fractional_qRBCb)
        # blood
        fractional_a_blood.append(fractional_flow_a)
        fractional_b_blood.append(fractional_flow_b)

        hemat_parent_plot.append(hemat_par)

        if hemat_a >= 1 or hemat_b >= 1:
            print(hemat_a)
            print(hemat_b)
            sys.exit()
        return hemat_a, hemat_b

    def update_hd(self, flownetwork):
        """

        """
        flownetwork.fractional_a_qRBCs = []
        flownetwork.fractional_b_qRBCs = []
        flownetwork.fractional_a_blood = []
        flownetwork.fractional_b_blood = []
        flownetwork.hemat_parent_plot = []
        fractional_a_qRBCs = flownetwork.fractional_a_qRBCs
        fractional_b_qRBCs = flownetwork.fractional_b_qRBCs
        fractional_a_blood = flownetwork.fractional_a_blood
        fractional_b_blood = flownetwork.fractional_b_blood
        hemat_parent_plot = flownetwork.hemat_parent_plot
        # diameter
        diameter = copy.copy(flownetwork.diameter)
        # solved by the system
        pressure = copy.copy(flownetwork.pressure)
        # number of nodes
        nr_of_vs = copy.copy(flownetwork.nr_of_vs)
        # edge list
        edge_list = copy.copy(flownetwork.edge_list)
        # base of arrays
        pressure_node = np.zeros((nr_of_vs, 2))

        rbc_balance = 0

        if pressure is None:
            flownetwork.hd = copy.copy(flownetwork.ht)
        else:
            flow = np.abs(copy.copy(flownetwork.flow_rate))

            # [pressure][node] in a single array
            for pres in range(0, nr_of_vs):
                pressure_node[pres] = np.array([pressure[pres], pres])

            # ordered in base of pressure [pressure][node]
            pressure_node = pressure_node[pressure_node[:, 0].argsort()[::-1]]
            ordered_node = pressure_node[:, 1]
            # iterate over the nodes, ordered by pressure
            for node in pressure_node:
                # position of the node connected
                edge_connected_position = []
                # node connceted
                node_connected = []
                edge_daughter = []
                # position of the parent edge
                parent_edge = []
                edge_connected = []
                # check if we are in a boundary node
                if node[1] == self._PARAMETERS["hexa_boundary_vertices"][0]:
                    # look for the daughters
                    for edge in range(0, len(edge_list)):
                        if edge_list[edge][0] == node[1] or edge_list[edge][1] == node[1]:
                            edge_connected_position.append(edge)
                            edge_connected.append(edge_list[edge])
                            # all node connected
                            if edge_list[edge][0] == node[1]:
                                edge_daughter.append(edge_list[edge][1])
                            else:
                                edge_daughter.append(edge_list[edge][0])
                    # DAUGHTERS
                    # Check witch type of daughters (bifurcation)
                    match len(edge_daughter):
                        case 3:
                            daughter_a, daughter_b, daughter_c = edge_daughter[0], edge_daughter[1], edge_daughter[2]
                            n_daughter = 3
                        case 2:  # bifurcation (ø<)
                            daughter_a, daughter_b = edge_daughter[0], edge_daughter[1]
                            n_daughter = 2
                        case 1:  # one daughter (ø-)
                            single_daughter = edge_daughter[0]
                            n_daughter = 1
                        case _:  # no daughters
                            sys.exit()
                    match n_daughter:
                        case 1:
                            flow_parent = flow[single_daughter]
                            # in case the flows are not the same
                            flownetwork.hd[single_daughter] = self._PARAMETERS["boundary_hematocrit"]
                            rbc_balance += self.qRCS(flownetwork, 2, flow_parent, flow[single_daughter], None, None,
                                                     self._PARAMETERS["boundary_hematocrit"],
                                                     flownetwork.hd[single_daughter], None, None)
                        case 2:  # (<)
                            diameter_parent = np.average([diameter[daughter_a], diameter[daughter_b]])
                            flow_parent = np.average([flow[daughter_a], flow[daughter_b]])
                            flownetwork.hd[daughter_a], flownetwork.hd[daughter_b] = self.get_erythrocyte_fraction(
                                self._PARAMETERS["boundary_hematocrit"],
                                diameter_parent,
                                diameter[daughter_a],
                                diameter[daughter_b],
                                flow[daughter_a], flow_parent, flow[daughter_b],
                                fractional_a_qRBCs,
                                fractional_b_qRBCs, fractional_a_blood, fractional_b_blood, hemat_parent_plot)
                            rbc_balance += self.qRCS(flownetwork, 1, flow_parent, flow[daughter_a], flow[daughter_b],
                                                     None,
                                                     self._PARAMETERS["boundary_hematocrit"],
                                                     flownetwork.hd[daughter_a],
                                                     flownetwork.hd[daughter_b], None)
                        case 3:  # (E)
                            flow_parent = np.average([flow[daughter_a], flow[daughter_b], flow[daughter_c]])

                            flownetwork.hd[daughter_a] = (flow[daughter_a] * self._PARAMETERS[
                                "boundary_hematocrit"]) / flow_parent
                            flownetwork.hd[daughter_b] = (flow[daughter_b] * self._PARAMETERS[
                                "boundary_hematocrit"]) / flow_parent
                            flownetwork.hd[daughter_c] = (flow[daughter_c] * self._PARAMETERS[
                                "boundary_hematocrit"]) / flow_parent

                        # TODO: implement the case of converging bifurcation, expected with multiple inflows

                else:
                    # create the list of the edge connected with that node
                    for edge in range(0, len(edge_list)):
                        # save the position of the edge in the edge_list
                        if edge_list[edge][0] == node[1] or edge_list[edge][1] == node[1]:
                            edge_connected_position.append(edge)
                            edge_connected.append(edge_list[edge])
                            # all node connected
                            if edge_list[edge][0] == node[1]:
                                node_connected.append(edge_list[edge][1])
                            else:
                                node_connected.append(edge_list[edge][0])

                        # find which is the parent node and the daughters one
                    for on in ordered_node:
                        # nodo nella lista oridnata è il nodo considerato, non si trova nella lista connected
                        # ho trovato dove sia nella lista ordinata
                        if on == node[1]:
                            for element in edge_connected_position:
                                if not np.isin(element, parent_edge):
                                    edge_daughter.append(element)
                            break
                        elif np.isin(on, edge_connected):
                            # connected node è nella lista dei connessi ne ho trovato uno nella lista quindi è un
                            # parent lo tolgo dalla lista dei nodi connessi e lo inserisco nella lista di nodi
                            # parent
                            for i in range(0, (len(edge_connected))):
                                if np.isin(on, edge_connected[i]):
                                    # positizione del parent edge
                                    parent_edge.append(edge_connected_position[i])

                    # DAUGHTERS
                    # Check witch type of daughters (bifurcation)
                    match len(edge_daughter):
                        case 1:  # one daughter (ø-)
                            single_daughter = edge_daughter[0]
                            n_daughter = 1
                        case 2:  # bifurcation (ø<)
                            daughter_a, daughter_b = edge_daughter[0], edge_daughter[1]
                            # if flow[daughter_b] < flow[daughter_a]:
                            #     daughter_a, daughter_b = copy.copy(edge_daughter[1]), copy.copy(edge_daughter[0])
                            n_daughter = 2
                        case 3:  # trifocation
                            daughter_a, daughter_b, daughter_c = edge_daughter[0], edge_daughter[1], edge_daughter[2]
                            n_daughter = 3
                        case _:  # no daughters
                            n_daughter = 0

                    # PARENTS
                    # Check witch type of parents
                    match len(parent_edge):
                        case 1:  # single parent (-ø)
                            parent = parent_edge[0]
                            n_parent = 1
                        case 2:  # two parents (>ø)
                            parent_a, parent_b = parent_edge[0], parent_edge[1]
                            n_parent = 2
                        case 3:  # trifocation
                            parent_a, parent_b, parent_c = parent_edge[0], parent_edge[1], parent_edge[2]
                            n_parent = 3
                        case _:
                            n_parent = 0

                    match n_parent, n_daughter:
                        # one parent first case
                        case 0, _:
                            print("Node " + str(node) + " has no parent")

                        # one parent and two daughter (-<)
                        case 1, 2:
                            flownetwork.hd[daughter_a], flownetwork.hd[daughter_b] = self.get_erythrocyte_fraction(
                                flownetwork.hd[parent],
                                diameter[parent], diameter[daughter_a], diameter[daughter_b],
                                flow[daughter_a], flow[parent], flow[daughter_b],
                                fractional_a_qRBCs, fractional_b_qRBCs, fractional_a_blood, fractional_b_blood,
                                hemat_parent_plot)
                            rbc_balance += self.qRCS(flownetwork, 1, flow[parent], flow[daughter_a], flow[daughter_b],
                                                     None, flownetwork.hd[parent], flownetwork.hd[daughter_a],
                                                     flownetwork.hd[daughter_b], None)
                        # one parent and one daughter (-ø-)
                        case 1, 1:
                            if flow[parent] != flow[single_daughter]:
                                # extract the RBCs in the parent
                                qRBCp = flownetwork.hd[parent] * flow[parent]
                                flownetwork.hd[single_daughter] = qRBCp / flow[single_daughter]
                                rbc_balance += self.qRCS(flownetwork, 2, flow[parent], flow[single_daughter], None,
                                                         None, flownetwork.hd[parent], flownetwork.hd[single_daughter],
                                                         None, None)
                                # same flows
                            else:
                                flownetwork.hd[single_daughter] = flownetwork.hd[parent]
                                rbc_balance += self.qRCS(flownetwork, 2, flow[parent], flow[single_daughter], None,
                                                         None, flownetwork.hd[parent], flownetwork.hd[single_daughter],
                                                         None, None)
                        # two parents and one daughter (>-)
                        case 2, 1:
                            flownetwork.hd[single_daughter] = ((flow[parent_a] * flownetwork.hd[parent_a]) + (
                                    flow[parent_b] * flownetwork.hd[parent_b])) / (
                                                                      np.abs(flow[parent_a]) + np.abs(flow[parent_b]))

                            rbc_balance += self.qRCS(flownetwork, 3, flow[single_daughter], flow[parent_a],
                                                     flow[parent_b], None, flownetwork.hd[single_daughter],
                                                     flownetwork.hd[parent_a], flownetwork.hd[parent_b], None)
                        # three parents and one daughter (∃ø-)
                        case 3, 1:
                            flownetwork.hd[single_daughter] = ((flow[parent_a] * flownetwork.hd[parent_a]) + (
                                    flow[parent_b] * flownetwork.hd[parent_b]) + (flow[parent_c] * flownetwork.hd[
                                parent_c])) / (flow[parent_a] + flow[parent_b] + flow[parent_c])

                            rbc_balance += self.qRCS(flownetwork, 4, flow[single_daughter], flow[parent_a],
                                                     flow[parent_b], flow[parent_c], flownetwork.hd[single_daughter],
                                                     flownetwork.hd[parent_a], flownetwork.hd[parent_b],
                                                     flownetwork.hd[parent_c])
                        # three parents and one daughter (∃ø<)
                        case 3, 2:
                            diameter_parent = np.average([diameter[daughter_a], diameter[daughter_b]],
                                                         diameter[daughter_c])
                            flow_parent = np.average([flow[daughter_a], flow[daughter_b]], flow[daughter_c])
                            hematocrit_parent = np.average([flownetwork.hd[daughter_a], flownetwork.hd[daughter_b]],
                                                           flownetwork.hd[daughter_c])

                            flownetwork.hd[daughter_a], flownetwork.hd[daughter_b] = self.get_erythrocyte_fraction(
                                hematocrit_parent, diameter_parent, diameter[daughter_a], diameter[daughter_b],
                                flow[daughter_a], flow_parent, flow[daughter_b], fractional_a_qRBCs,
                                fractional_b_qRBCs, fractional_a_blood, fractional_b_blood, hemat_parent_plot)

                            rbc_balance += self.qRCS(flownetwork, 5, flow_parent, flow[daughter_a], flow[daughter_b],
                                                     flow[daughter_c], hematocrit_parent,
                                                     flownetwork.hd[daughter_a], flownetwork.hd[daughter_b],
                                                     flownetwork.hd[daughter_c])
                        # One parent and three daughters (-øE)
                        case 1, 3:
                            flownetwork.hd[daughter_a] = (flow[daughter_a] * flownetwork.hd[parent]) / flow[parent]
                            flownetwork.hd[daughter_b] = (flow[daughter_b] * flownetwork.hd[parent]) / flow[parent]
                            flownetwork.hd[daughter_c] = (flow[daughter_c] * flownetwork.hd[parent]) / flow[parent]

                            # TODO: qRBS balance
                        # Two parents and three daughters (>øE)
                        case 2, 3:
                            flow_parent = np.average([flow[parent_a], flow[parent_b]])
                            hematocrit_parent = np.average([flownetwork.hd[parent_a], flownetwork.hd[parent_b]])

                            flownetwork.hd[daughter_a] = (flow[daughter_a] * hematocrit_parent) / flow_parent
                            flownetwork.hd[daughter_b] = (flow[daughter_b] * hematocrit_parent) / flow_parent
                            flownetwork.hd[daughter_c] = (flow[daughter_c] * hematocrit_parent) / flow_parent

                            # TODO: qRBS balance
                        # Three parents and three daughters (∃øE)
                        case 3, 3:

                            flow_parent = np.average([flow[daughter_a], flow[daughter_b]], flow[daughter_c])
                            hematocrit_parent = np.average([flownetwork.hd[daughter_a], flownetwork.hd[daughter_b]],
                                                           flownetwork.hd[daughter_c])

                            flownetwork.hd[daughter_a] = (flow[daughter_a] * hematocrit_parent) / flow_parent
                            flownetwork.hd[daughter_b] = (flow[daughter_b] * hematocrit_parent) / flow_parent
                            flownetwork.hd[daughter_c] = (flow[daughter_c] * hematocrit_parent) / flow_parent

                            # TODO: qRBS balance

            print("Check RBCs balance: ...")
            if rbc_balance > 0:
                sys.exit("Check RBCs balance: FAIL -->", rbc_balance)
            else:
                print("Check RBCs balance: DONE")
                util_display_graph(graph_creation(flownetwork), flownetwork.iteration, self._PARAMETERS, flownetwork)
