"""

To calculate the different hematocrit in the graph is needed to be calculated for each vessel

for this reason it has been added an iflow vessel and out flow vessul to the fra ph

"""

from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
from math import e

from source.bloodflowmodel.transmissibility import TransmissibilityVivoPries1996


class IterativeHematocrit(ABC):
    """
    Abstract base class for the implementations of iterative approach
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of Iterative.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS
        self.x_o_init = 1.12e-6  # micrometers
        self.A_o_init = 15.47e-6  # micrometers
        self.B_o_init = 8.13e-6  # micrometers

    @abstractmethod
    def get_erythrocyte_fraction(self, hemat_par, diam_par, diam_dgt_a, diam_dgt_b, flow_a):
        """
         to calculate the fraction of erythrocyte that goes in each daughter vessel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        pass

    @abstractmethod
    def update_hemtocrit_linear(self, flownetwork):
        pass


def _logit(x):
    return np.log(x / (1 - x))


class IterativeLorthois2011(IterativeHematocrit):

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

        # roba che devo capire come inserire
        # frazione di sangue la interpreto come la quantità di flusso presente all'interno del vessel

        # relationship used to calculate the fraction of erythrocyte
        if flow_a <= x_0:

            # hemat_a = 0
            # condition to not create unbalanced network
            hemat_a = 0.2*flow_a
            hemat_b = 0.8*flow_a
        elif flow_a >= 1 - x_0:
            # hemat_a = 1
            # condition to not create unbalanced network
             hemat_a = 0.8*flow_a
             hemat_b = 0.2*flow_a

        else:
            logit_F_Q_a_e = A + B * _logit((flow_a - x_0) / (1 - x_0))
            hemat_a = pow(e, logit_F_Q_a_e) / (1 + pow(e, logit_F_Q_a_e))
            hemat_b = 1 - hemat_a

        return hemat_a, hemat_b

    def update_hemtocrit_linear(self, flownetwork):

        # initial supposed hematocrit
        ht_init = self._PARAMETERS["ht_constant"]
        # hematocrit in each vessel
        hematocrit = flownetwork.hd
        # diameter
        diameter = flownetwork.diameter
        # solved by the system
        pressure = flownetwork.pressure
        # number of nodes
        nr_of_vs = flownetwork.nr_of_vs
        # number of edges
        nr_of_es = flownetwork.nr_of_es
        # edge list
        edge_list = flownetwork.edge_list
        # flow
        flow = flownetwork.flow_rate
        # base of arrays
        pressure_node = np.zeros((nr_of_vs, 2))
        # pressure_node[0] = np.array([pressure[0], pressure.index(max(pressure))])

        # TODO hidden
        diameter_hidden = 5e-6
        hematocrit_hidden = 5e-6

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
                if edge_list[edge][0] == node[1]:
                    edge_list_node.append(edge)
                # all the edges connected to the node (parents)
                if edge_list[edge][1] == node[1]:
                    parent_list.append(edge_list[edge][0])

            # DAUGHTERS
            # Check witch type of daughters (bifurcation)
            match len(edge_list_node):
                case 2:  # bifurcation (ø<)
                    daughter_a, daughter_b = edge_list_node[0], edge_list_node[1]
                case 1:  # one daughter (ø-)
                    single_daughter = edge_list_node[0]
                    two_daughters = False
                case _:  # multiple daughters TODO: implement
                    pass

            # PARENTS
            # Check witch type of parents
            match len(parent_list):
                case 1:  # single parent (-ø)
                    parent = parent_list[0]
                case 2:  # two parents (>ø)
                    parent_a, parent_b = parent_list[0], parent_list[1]
                    one_parent = False
                case _:  # multiple parents >2 TODO: implement
                    parent = 0

            # one parent and two daughter (-<)
            if two_daughters and one_parent:
                flownetwork.hd[daughter_a], flownetwork.hd[daughter_b] = self.get_erythrocyte_fraction(
                    hematocrit[parent],
                    diameter[parent],
                    diameter[daughter_a],
                    diameter[daughter_b],
                    flow[daughter_a])

            #  one parent and one daughter (-ø-)
            elif one_parent:
                flownetwork.hd[single_daughter] = hematocrit[parent]

            # two parent and one daughter (>-)
            else:
                flownetwork.hd[single_daughter] = ((flow[parent_a] * hematocrit[parent_a]) + (
                            flow[parent_b] * hematocrit[parent_b])) / (flow[parent_a] + flow[parent_b])

        # update of the pressure based on the new hematocrit
        TransmissibilityVivoPries1996.update_transmiss(None, flownetwork)
        flownetwork.update_blood_flow()
