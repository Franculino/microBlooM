"""

To calculate the different hematocrit in the graph is needed to be calculated for each vessel

for this reason it has been added an iflow vessel and out flow vessul to the fra ph

"""

from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
from math import e


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
    def _get_erythrocyte_fraction(self, hemat_par, diam_par, diam_dgt_a, diam_dgt_b, hemat_a, hemat_b):
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

    def _get_erythrocyte_fraction(self, hemat_par, diam_par, diam_dgt_a, diam_dgt_b, hemat_a, hemat_b):
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
        fractional_blood_a, logit_F_Q_a_e = 0, 0

        # relationship used to calculate the fraction of erythrocyte
        if fractional_blood_a <= x_0:
            # hemat_a = 0
            # condition to not create unbalanced network
            hemat_a = 0.2
            hemat_b = 0.8
        elif fractional_blood_a >= 1 - x_0:
            # hemat_a = 1
            # condition to not create unbalanced network
            hemat_a = 0.8
            hemat_b = 0.2
        else:
            logit_F_Q_a_e = A + B * _logit((hemat_a - x_0) / (1 - x_0))
            hemat_a = pow(e, logit_F_Q_a_e) / (1 + pow(e, logit_F_Q_a_e))
            hemat_b = 1 - hemat_a

        return hemat_a, hemat_b

    def update_hemtocrit_linear(self, flownetwork):

        # initial supposed hematocrit
        ht_init = self._PARAMETERS["ht_constant"]
        # solved by the system
        pressure = flownetwork.pressure
        # number of nodes
        nr_of_vs = flownetwork.nr_of_vs
        # number of edges
        nr_of_es = flownetwork.nr_of_es
        # edge list
        edge_list = flownetwork.edge_list
        # base of arrays
        pressure_node = np.zeros((nr_of_vs, 2))
        # pressure_node[0] = np.array([pressure[0], pressure.index(max(pressure))])

        # TODO hidden
        diameter_hidden = 5e-6
        hematocrit_hidden = 5e-6

        # [pressure][node] in a single array
        for pres in range(0, nr_of_vs):
            pressure_node[pres] = np.array([pressure[pres], pres])

        # orderered in base of pressure [pressure][node]
        pressure_node = pressure_node[np.argsort(pressure_node[:, 0][::-1])]

        # connection of each node
        connection = {}
        for group in edge_list:
            for member in group:
                connection.setdefault(member, set()).update((set(group) - {member}))


        # first pressure node is the highest
        # check witch node and then the connection
        first = connection[pressure_node[0][1]]

        """
        - update dei due hematocriti
        - ricalcolo mu
        - ricalcolo G
        - ricaclolo P con sistema
        - aggiorno lista delle pressioni, non tocco l'iniziale perchè è boundary
        """

        pass
