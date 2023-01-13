# todo: class for handling the mapping from alpha prime to alpha and vis-verca
#  2 options: No mapping, tanh mapping
#  This is a construction site

from abc import ABC, abstractmethod
from types import MappingProxyType

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
    def get_d_alpha_d_alpha_prime(self):
        pass

    @abstractmethod
    def update_alpha_from_alpha_prime(self):
        pass