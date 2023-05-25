import sys
from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np


class Transmissibility(ABC):
    """
    Abstract base class for the implementations related to calculating the transmissibility
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of Transmissibility.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    def _get_transmiss_poiseuille(self, flownetwork):
        """
        Calculate the transmissibility based on poiseuille's law. Does not take red blood cells into account.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Transmissibility in every edge
        :rtype: 1d numpy array
        """
        diameter = flownetwork.diameter
        length = flownetwork.length
        mu_plasma = self._PARAMETERS["mu_plasma"]

        # Poiseuille's law to calculate transmissibility without red blood cell influence.
        return np.pi * np.power(diameter, 4) / (128 * mu_plasma * length)

    @abstractmethod
    def update_transmiss(self, flownetwork):
        """
        Update the transmissibility and mu_rel in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class TransmissibilityPoiseuille(Transmissibility):
    """
    Class for calculating the transmissibility without red blood cells (Newtonian flow)
    """

    def update_transmiss(self, flownetwork):
        """
        Update the transmissibility in flownetwork based on poiseuille's law. Does not take red blood cells into
        account. Mu_rel is one everywhere
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.mu_rel = np.ones(flownetwork.nr_of_es)
        flownetwork.transmiss = self._get_transmiss_poiseuille(flownetwork)  # implementation from base class


class TransmissibilityVitroPries1992(Transmissibility):
    """
    Class for calculating the transmissibility with red blood cells. Also calculates mu_rel. The impact of red blood
    cells is considered by the empirical in vitro equations by Pries, Neuhaus, Gaehtgens (1992).
    """

    def update_transmiss(self, flownetwork):
        """
        Update the transmissibility in flownetwork based on poiseuille's law and the empirical in vitro equations
        by Pries, Neuhaus, Gaehtgens (1992).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        transmiss_poiseuille = self._get_transmiss_poiseuille(flownetwork)  # Transmissibility without red blood cells.

        # Relative viscosity based on Pries, Neuhaus, Gaehtgens (1992)
        diameter_um = 1.e6 * flownetwork.diameter  # Diameter in micro meters
        hd = flownetwork.hd

        C = (0.8 + np.exp(-0.075 * diameter_um)) * (-1. + 1. / (1. + 1.e-11 * np.power(diameter_um, 12.))) + 1. / (
                1 + 1.e-11 * np.power(diameter_um, 12.))  # Eq.(4) in paper
        mu_rel_45 = 220 * np.exp(-1.3 * diameter_um) + 3.2 - 2.44 * np.exp(
            -0.06 * np.power(diameter_um, 0.645))  # Eq.(2) in paper

        mu_rel = 1 + (mu_rel_45 - 1) * (np.power((1 - hd), C) - 1) / (np.power((1. - 0.45), C) - 1)  # Eq.(7) in paper

        flownetwork.mu_rel = mu_rel
        flownetwork.transmiss = transmiss_poiseuille / mu_rel  # Update transmissibility.


class TransmissibilityVitroPries2005(Transmissibility):
    """
    Class for calculating the transmissibility with red blood cells. Also calculates mu_rel. The impact of red blood
    cells is considered by the empirical in vitro equations by Pries and Secomb (2005).
    """

    def update_transmiss(self, flownetwork):
        """
        Update the transmissibility in flownetwork based on poiseuille's law and the empirical in vitro equations
        by Pries and Secomb (2005).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        transmiss_poiseuille = self._get_transmiss_poiseuille(flownetwork)  # Transmissibility without red blood cells.

        # Relative viscosity based on Pries and Secomb (2005).
        diameter_um = 1.e6 * flownetwork.diameter  # Diameter in micro meters
        hd = flownetwork.hd

        C = (0.8 + np.exp(-0.075 * diameter_um)) * (-1 + 1. / (1. + 1.e-11 * np.power(diameter_um, 12.))) \
            + 1. / (1. + 1.e-11 * np.power(diameter_um, 12.))
        mu_rel_45 = 220 * np.exp(-1.3 * diameter_um) + 3.2 - 2.44 * np.exp(-0.06 * np.power(diameter_um, 0.645))
        mu_rel = 1. + (mu_rel_45 - 1.) * ((np.power((1. - hd), C) - 1.) / (np.power((1. - 0.45), C) - 1.))

        flownetwork.mu_rel = mu_rel
        flownetwork.transmiss = transmiss_poiseuille / mu_rel  # Update transmissibility.


class TransmissibilityVivoPries1996(Transmissibility):
    """
    Class for calculating the transmissibility taking into account the Fahraeus–Lindquist effect.
    The following in vivo phenomenological relationship describing the variations of apparent viscosity as a function of
    vessel diameter (expressed in micrometers) and discharge hematocrit.
    """

    def update_transmiss(self, flownetwork):
        mu_plasma = 0.0012
        # Diameter in micro meters
        diameter_um = flownetwork.diameter
        # diameter_um = [i * 1.e6 for i in flownetwork.diameter]
        # discharging hematocrit
        length = flownetwork.length
        hd = flownetwork.hd
        i = 0

        for diameter in diameter_um:
            if flownetwork.transmiss is None:
                flownetwork.transmiss = np.zeros(flownetwork.nr_of_es)
            # viscosity for discharging hematocrit of 0.45
            mu_rel_45 = 6 * np.exp(-0.085 * diameter) + 3.2 - 2.44 * np.exp(-0.06 * np.power(diameter, 0.645))
            # coefficient C
            C = (0.8 + np.exp(-0.075 * diameter)) * (-1 + 1. / (1. + 1e-10 * np.power(diameter, 12.))) \
                + 1. / (1. + 1.e-10 * np.power(diameter, 12.))
            # variation of apparent viscosity, mu_rel = mu relative to this vessel
            mu_rel = mu_plasma * (
                    1. + (mu_rel_45 - 1.) * ((np.power((1. - hd[i]), C) - 1.) / (np.power((1. - 0.45), C) - 1.)) * (
                np.power((diameter / (diameter - 1.1)), 2)))

            flownetwork.transmiss[i] = np.pi * np.power(diameter, 4) / (128 * mu_rel * flownetwork.length[i])
            i += 1
