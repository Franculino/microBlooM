# todo: class for handling the mapping from alpha prime to alpha and vis-verca
#  2 options: No mapping, tanh mapping
#  This is a construction site

from abc import ABC, abstractmethod
from types import MappingProxyType

import numpy as np


class AlphaMapping(ABC):
    """
    Abstract base class for the implementations the mapping from a pseudo parameter alpha_prime to alpha
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of AlphaMapping.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def update_gradient_alpha_prime(self, inversemodel):
        """
        Bla
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """

    @abstractmethod
    def update_alpha_from_alpha_prime(self, inversemodel):
        """
        Bla
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """


class AlphaMappingLinear(AlphaMapping):
    """
    Class mapping from a pseudo parameter alpha_prime to alpha. Here, no mapping (alpha = alpha_prime)
    """

    def update_gradient_alpha_prime(self, inversemodel):
        inversemodel.gradient_alpha_prime = inversemodel.gradient_alpha

    def update_alpha_from_alpha_prime(self, inversemodel):
        inversemodel.alpha = inversemodel.alpha_prime


class AlphaMappingTanh(AlphaMapping):
    """
    Class mapping from a pseudo parameter alpha_prime to alpha. Here, no mapping (alpha = alpha_prime)
    """

    def update_gradient_alpha_prime(self, inversemodel):
        delta = inversemodel.edge_param_pm_range
        phi = inversemodel.phi
        alpha_prime = inversemodel.alpha_prime

        d_alpha_d_alpha_prime = delta*phi*(1.-np.square(np.tanh(phi*(alpha_prime-1.))))

        inversemodel.gradient_alpha_prime = inversemodel.gradient_alpha * d_alpha_d_alpha_prime

    def update_alpha_from_alpha_prime(self, inversemodel):
        delta = inversemodel.edge_param_pm_range
        phi = inversemodel.phi
        alpha_prime = inversemodel.alpha_prime

        inversemodel.alpha = 1. + delta * np.tanh(phi * (alpha_prime - 1.))