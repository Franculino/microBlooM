from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class ReadDistensibilityParameters(ABC):
    """
    Abstract base class for the implementations ...
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of ReadParameters.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def read(self, distensibility, flownetwork):
        """
        Import...
        :param distensibility: inverse model object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class ReadDistensibilityParametersFromFile(ReadDistensibilityParameters):
    """
    Class for importing the parameter space related to edges
    """

    def read(self, distensibility, flownetwork):
        """
        Import...
        :param distensibility: inverse model object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # Todo: import from file. This is only temporarily here
        distensibility.e_modulus = np.ones(flownetwork.nr_of_es) * 2.6e5
        distensibility.wall_thickness = flownetwork.diameter * .1
        distensibility.can_adapt = np.array([True]*flownetwork.nr_of_es)

        # import pandas as pd
        # # Extract file path of target values.
        # path_edge_parameterspace = self._PARAMETERS["csv_path_edge_parameterspace"]
        #
        # # Read file with pandas
        # df_parameter_space = pd.read_csv(path_edge_parameterspace)
        #
        # # Sort prescribed edge ids with parameters according to ascending edge ids.
        # df_parameter_space = df_parameter_space.sort_values('edge_param_eid')
        # # Check for duplicate eids
        # if True in df_parameter_space.duplicated(subset=['edge_param_eid']).to_numpy():
        #     sys.exit("Error: Duplicate edge id in parameter space definition.")


