from abc import ABC, abstractmethod
from types import MappingProxyType
import sys
import numpy as np
from math import e
import source.bloodflowmodel.transmissibility as transmissibility
import source.bloodflowmodel.pressure_flow_solver as pressureflowsolver
import source.bloodflowmodel.build_system as buildsystem
import source.bloodflowmodel.rbc_velocity as rbc_velocity


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
        self.x_o_init = 1.12e-6  # micrometers
        self.A_o_init = 15.47e-6  # micrometers
        self.B_o_init = 8.13e-6  # micrometers

    @abstractmethod
    def update_hd(self, flownetwork):
        """
        Abstract method to update the discharge haematocrit in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def qRCS(self, flownetwork, case, flow_single, flow_a, flow_b, hemat_single, hemat_a, hemat_b):
        """
        check the RBC balance
        qRBC = q * Hdt

        Massbalances:
        q_p1 + q_p2 = q_d,
        qRBC_p1 + qRBC_p2 = qRBC_d1,
        """


    def _logit(self, x):
        return np.log(x / (1 - x))


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
        model is based on the empirical in vitro functions by Pries, Neuhaus, Gaehtgens (1992).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        ht = flownetwork.ht  # Tube haematocrit
        diameter_um = 1.e6 * flownetwork.diameter  # Vessel diameter in micro meters

        x_tmp = 1. + 1.7 * np.exp(-0.35 * diameter_um) - 0.6 * np.exp(-0.01 * diameter_um)  # Eq. (9) in paper
        x_bound = np.copy(x_tmp)
        x_bound[x_tmp > 0.99] = 0.99  # Bound x to values < 1. Equation in paper is only valid for x < 1.

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
        model is based on the empirical in vitro functions by Pries and Secomb (2005).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        ht = flownetwork.ht  # Tube haematocrit
        diameter_um = 1.e6 * flownetwork.diameter  # Vessel diameter in micro meters

        x_tmp = 1. + 1.7 * np.exp(-0.415 * diameter_um) - 0.6 * np.exp(-0.011 * diameter_um)  # From Eq.(1) in paper
        x_bound = np.copy(x_tmp)
        x_bound[x_tmp > 0.99] = 0.  # Bound x to values < 1. Equation in paper is only valid for x < 1.

        hd = -x_bound / (2 - 2 * x_bound) + np.sqrt(
            np.square(x_bound / (2 - 2 * x_bound)) + ht / (1 - x_bound))
        hd[x_tmp > 0.99] = ht[x_tmp > 0.99]  # For very small and very large diameters: set ht=hd

        flownetwork.hd = hd  # Update discharge haematocrit


class DischargeHaematocritLorthois2011(DischargeHaematocrit):

    def qRCS(self, flownetwork, case, flow_single, flow_a, flow_b, hemat_single, hemat_a, hemat_b):
        """
        check the RBC balance
        qRBC = q * Hdt

        Massbalances:
        q_p1 + q_p2 = q_d,
        qRBC_p1 + qRBC_p2 = qRBC_d1,
        """
        tollerance = 1.00E-05
        match case:
            case 2:  # one parent and two daughter (-<)
                flow_parent = flow_a + flow_b
                if np.abs(flow_single - flow_parent) <= tollerance:  # check same flow
                    qRBC_a = flow_a * hemat_a
                    qRBC_b = flow_b * hemat_b
                    qRBC_p = flow_single * hemat_single
                    qRBC_p_to_check = qRBC_a + qRBC_b
                    if np.abs(qRBC_p - qRBC_p_to_check) <= tollerance:  # check same RBCs
                        RBCbalance = 0
                    else:
                        RBCbalance = 1
                else:
                    RBCbalance = 1
            case 3:  # one parent and one daughter (-ø-)
                if np.abs(flow_single - flow_a) <= tollerance:  # check same flow
                    qRBC_single = flow_single * hemat_single
                    qRBC_a = flow_a * hemat_a
                    if np.abs(qRBC_single - qRBC_a) <= tollerance:
                        RBCbalance = 0
                    else:
                        RBCbalance = 1
                else:
                    RBCbalance = 1

                pass
            case 4:  # two parent and one daughter (>-)
                flow_daughter_to_check = flow_a + flow_b

                if np.abs(flow_single - flow_daughter_to_check) <= tollerance:  # check same flow
                    qRBC_a = flow_a * hemat_a
                    qRBC_b = flow_b * hemat_b
                    qRBC_d = flow_single * hemat_single
                    qRBC_d_to_check = qRBC_a + qRBC_b
                    if np.abs(qRBC_d - qRBC_d_to_check) <= tollerance:
                        RBCbalance = 0
                    else:
                        # print(">-", qRBC_d, qRBC_d_to_check)
                        RBCbalance = 1
                else:
                    # print(">-", flow_single, flow_daughter_to_check)
                    #   print(flow_single is flow_daughter_to_check)
                    RBCbalance = 1

            case _:  # no daughter
                print("pass case")
                pass
        return RBCbalance

    def get_erythrocyte_fraction(self, hemat_par, diam_par, diam_dgt_a, diam_dgt_b, flow_a):
        """
        to calculate the fraction of erythrocyte that goes in each daugheter vessel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        x_0 = self.x_o_init * (1 - hemat_par) / diam_par
        A = - self.A_o_init * (
                (pow(diam_dgt_a, 2) - pow(diam_dgt_b, 2)) / (pow(diam_dgt_a, 2) + pow(diam_dgt_b, 2))) * (
                    1 - hemat_par) / diam_par

        B = 1 + self.B_o_init * (1 - hemat_par) / diam_par

        # relationship used to calculate the fraction of erythrocyte
        if flow_a <= x_0:

            # hemat_a = 0
            # condition to not create unbalanced network
            hemat_a = 0.2 * flow_a
            hemat_b = 0.8 * flow_a
        elif flow_a >= 1 - x_0:
            # hemat_a = 1
            # condition to not create unbalanced network
            hemat_a = 0.8 * flow_a
            hemat_b = 0.2 * flow_a

        else:
            logit_F_Q_a_e = A + B * self._logit((flow_a - x_0) / (1 - x_0))
            hemat_a = pow(e, logit_F_Q_a_e) / (1 + pow(e, logit_F_Q_a_e))
            hemat_b = 1 - hemat_a

        return hemat_a, hemat_b

    def update_hd(self, flownetwork):

        # initial supposed hematocrit
        ht_init = self._PARAMETERS["ht_constant"]
        # diameter
        diameter = flownetwork.diameter
        # solved by the system
        pressure = flownetwork.pressure
        # number of nodes
        nr_of_vs = flownetwork.nr_of_vs
        # hematocrit
        hematocrit = flownetwork.hd
        # edge list
        edge_list = flownetwork.edge_list
        # flow
        flow = flownetwork.flow_rate
        # base of arrays
        pressure_node = np.zeros((nr_of_vs, 2))
        # pressure_node[0] = np.array([pressure[0], pressure.index(max(pressure))])

        rbc_balance = 0
        # TODO hidden
        diameter_hidden = 5e-6
        hematocrit_hidden = 5e-6

        if pressure is None:
            flownetwork.hd = flownetwork.ht

        else:
            # [pressure][node] in a single array
            for pres in range(0, nr_of_vs):
                pressure_node[pres] = np.array([pressure[pres], pres])

            # ordered in base of pressure [pressure][node]
            pressure_node = pressure_node[np.argsort(pressure_node[:, 0][::-1])]

            # iterate over the nodes, ordered by pressure
            for node in pressure_node:
                # True: one parent
                one_parent = True
                # True: two daughters
                two_daughters = True
                edge_list_node = []
                parent_list = []
                for edge in range(0, len(edge_list)):
                    # all the edges connected to the node (daughters)
                    # save the positional edge
                    if edge_list[edge][0] == node[1]:
                        edge_list_node.append(edge)
                    # all the edges connected to the node (parents)
                    if edge_list[edge][1] == node[1]:
                        # print(edge_list[edge])
                        parent_list.append(edge)

                # DAUGHTERS
                # Check witch type of daughters (bifurcation)
                match len(edge_list_node):
                    case 2:  # bifurcation (ø<)
                        daughter_a, daughter_b = edge_list_node[0], edge_list_node[1]
                        two_daughters = True
                    case 1:  # one daughter (ø-)
                        single_daughter = edge_list_node[0]
                        two_daughters = False
                    case _:  # no daughters
                        single_daughter = None
                        pass

                # PARENTS
                # Check witch type of parents
                match len(parent_list):
                    case 1:  # single parent (-ø)
                        parent = parent_list[0]
                        one_parent = True
                    case 2:  # two parents (>ø)
                        parent_a, parent_b = parent_list[0], parent_list[1]
                        one_parent = False
                    case _:  # multiple parents >2 TODO: implement
                        parent = None

                # no parent and two daughter (-<)
                if parent is None:
                    flownetwork.hd[daughter_a], flownetwork.hd[daughter_b] = self.get_erythrocyte_fraction(
                        ht_init,
                        5e-06,
                        diameter[daughter_a],
                        diameter[daughter_b],
                        flow[daughter_a])
                    # print("-- non ---")
                    # print(flownetwork.hd[daughter_a], flownetwork.hd[daughter_b])
                    # print("-- non ---")

                # one parent and two daughter (-<)
                elif two_daughters and one_parent:
                    flownetwork.hd[daughter_a], flownetwork.hd[daughter_b] = self.get_erythrocyte_fraction(
                        flownetwork.hd[parent],
                        diameter[parent],
                        diameter[daughter_a],
                        diameter[daughter_b],
                        flow[daughter_a])
                    rbc_balance += self.qRCS(flownetwork, 2, flow[parent], flow[daughter_a], flow[daughter_b],
                                             flownetwork.hd[parent],
                                             flownetwork.hd[daughter_a], flownetwork.hd[daughter_b])

                #  one parent and one daughter (-ø-)
                elif one_parent:
                    # print("---- single---")
                    # print(flownetwork.hd[parent])
                    flownetwork.hd[single_daughter] = flownetwork.hd[parent]
                    # print(flownetwork.hd[single_daughter])
                    # print("---- single---")
                    rbc_balance += self.qRCS(flownetwork, 3, flow[parent], flow[single_daughter], None,
                                             flownetwork.hd[parent], flownetwork.hd[single_daughter], None)

                elif single_daughter is None:
                    pass

                # two parent and one daughter (>-)
                else:
                    flownetwork.hd[single_daughter] = ((flow[parent_a] * flownetwork.hd[parent_a]) + (
                            flow[parent_b] * flownetwork.hd[parent_b])) / (flow[parent_a] + flow[parent_b])
                    rbc_balance += self.qRCS(flownetwork, 4, flow[single_daughter], flow[parent_a], flow[parent_b],
                                             flownetwork.hd[single_daughter], flownetwork.hd[parent_a],
                                             flownetwork.hd[parent_b])

            print("Check RBCs balance: ...")
            if rbc_balance > 0:
                sys.exit("Check RBCs balance: FAIL -->", rbc_balance)
            else:
                print("Check RBCs balance: DONE")

