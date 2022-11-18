from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys
# todo: this is a construction site


class ParameterSpace(ABC):
    """
    Abstract base class for the implementations specific to the chosen parameter space
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of ParameterSpace.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def initialise_parameters(self, inversemodel, flownetwork):
        """
        Initialises the parameter space
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def get_dT_dalpha(self, inversemodel, flownetwork):
        """
        Computes the partial derivative of all edge transmissibilities with respect to the pseudo parameter alpha_prime
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :return: Derivative d Transmissibility / d alpha
        :rtype: 1d numpy array
        """


class ParameterSpaceRelativeDiameter(ParameterSpace):
    """
    Class for a parameter space that includes all vessel diameters relative to baseline
    """

    def initialise_parameters(self, inversemodel, flownetwork):
        """
        Initialises the parameter space, if relative diameters are tuned
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        inversemodel.diameter_base = flownetwork.diameter
        inversemodel.alpha = np.ones(flownetwork.nr_of_es)
        inversemodel.alpha_prime = np.ones(flownetwork.nr_of_es)

    def get_dT_dalpha(self, inversemodel, flownetwork):
        """
        Computes the partial derivative of all edge transmissibilities with respect to the pseudo parameter alpha
        dT/dalpha = dT / d diameter * diameter_base
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :return: Derivative d Transmissibility / d alpha
        :rtype: 1d numpy array
        """
        # Todo compute derivative
        return np.ones(flownetwork.nr_of_es) # todo: this has to be replaced by the actual derivative


class ParameterSpaceRelativeTransmissibility(ParameterSpace):
    """
    Class for a parameter space that includes all transmissibilities relative to baseline
    """

    def initialise_parameters(self, inversemodel, flownetwork):
        """
        Initialises the parameter space, if relative transmissibilities are tuned
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        inversemodel.transmiss_base = np.copy(flownetwork.transmiss)
        inversemodel.alpha = np.ones(flownetwork.nr_of_es)
        inversemodel.alpha_prime = np.ones(flownetwork.nr_of_es)

    def get_dT_dalpha(self, inversemodel, flownetwork):
        """
        Computes the partial derivative of all edge transmissibilities with respect to the pseudo parameter alpha
        dT/dalpha = T_base
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :return: Derivative d Transmissibility / d alpha
        :rtype: 1d numpy array
        """
        return np.ones(flownetwork.nr_of_es) * inversemodel.transmiss_base
