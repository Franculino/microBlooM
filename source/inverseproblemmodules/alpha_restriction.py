from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np


class AlphaRestriction(ABC):
    """
    Abstract base class for the implementations related to the mapping from a pseudo parameter alpha_prime to alpha, in
    order to restrict alpha to desired ranges.
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
        Updates the gradient with respect to alpha prime.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """

    @abstractmethod
    def update_alpha_from_alpha_prime(self, inversemodel):
        """
        Updates alpha based on alpha_prime.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """


class AlphaRestrictionLinear(AlphaRestriction):
    """
    Class mapping from a pseudo parameter alpha_prime to alpha. Here, alpha is not restricted and alpha = alpha_prime.
    """

    def update_gradient_alpha_prime(self, inversemodel):
        """
        Updates the gradient of f with respect to the pseudo parameter alpha_prime, if no mapping is used.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """
        inversemodel.gradient_alpha_prime = inversemodel.gradient_alpha

    def update_alpha_from_alpha_prime(self, inversemodel):
        """
        Sets alpha equal to alpha_prime (no mapping is used)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """
        inversemodel.alpha = inversemodel.alpha_prime


class AlphaMappingTanh(AlphaRestriction):
    """
    Class mapping from a pseudo parameter alpha_prime to alpha. Here, the parameter alpha is restricted
    to a range 1 +/- range
    """

    def update_gradient_alpha_prime(self, inversemodel):
        """
        Updates the gradient of f with respect to the pseudo parameter alpha_prime, if a tanh mapping is used.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """
        delta = inversemodel.edge_param_pm_range
        phi = inversemodel.phi
        alpha_prime = inversemodel.alpha_prime

        # derivative d alpha / d alpha_prime
        d_alpha_d_alpha_prime = delta * phi * (1. - np.square(np.tanh(phi * (alpha_prime - 1.))))
        # Update the gradient
        inversemodel.gradient_alpha_prime = inversemodel.gradient_alpha * d_alpha_d_alpha_prime

    def update_alpha_from_alpha_prime(self, inversemodel):
        """
        Updates the parameter alpha based on the pseudo parameter alpha_prime, if a tanh mapping is used.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """
        delta = inversemodel.edge_param_pm_range
        phi = inversemodel.phi
        alpha_prime = inversemodel.alpha_prime

        # Restrict alpha prime to 1 +- 3/phi (alpha_prime with alpha = 0.995*alpha_max) to not drift too far away
        # from the symmetry proint alpha_prime=1 and alpha=1. Otherwise, the gradient gets very small.
        alpha_prime[alpha_prime < 1. - 3./phi] = 1. - 3./phi
        alpha_prime[alpha_prime > 1. + 3. / phi] = 1. + 3. / phi

        # Update the parameter alpha
        inversemodel.alpha = 1. + delta * np.tanh(phi * (alpha_prime - 1.))