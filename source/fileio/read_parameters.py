from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class ReadParameters(ABC):
    """
    Abstract base class for the implementations related to importing the parameter space
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of ReadParameters.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def read(self, inversemodel, flownetwork):
        """
        Read parameter space
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class ReadParametersEdges(ReadParameters):
    """
    Class for importing the parameter space related to edges
    """

    def read(self, inversemodel, flownetwork):
        """
        Import edge ids of parameters and the corresponding allowable change (1 +- range)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        import pandas as pd
        # Extract file path of target values.
        path_edge_parameterspace = self._PARAMETERS["csv_path_edge_parameterspace"]

        # Read file with pandas
        df_parameter_space = pd.read_csv(path_edge_parameterspace)

        # Check for keyword all to include all edge ids into the parameter space.
        if df_parameter_space['edge_param_eid'][0] == 'all':
            # All edge ids are parameters
            inversemodel.edge_param_eid = np.arange(flownetwork.nr_of_es)
            # The range is identical for all parameters (corresponds to the first value in the column.
            inversemodel.edge_param_pm_range = np.ones(flownetwork.nr_of_es) * \
                                               df_parameter_space['edge_param_pm_range'][0]
        else:
            # Sort prescribed edge ids with parameters according to ascending edge ids.
            df_parameter_space = df_parameter_space.sort_values('edge_param_eid')
            # Check for duplicate eids
            if True in df_parameter_space.duplicated(subset=['edge_param_eid']).to_numpy():
                sys.exit("Error: Duplicate edge id in parameter space definition.")

            # Assign data to inversemodel object
            # Edge attributes
            inversemodel.edge_param_eid = df_parameter_space["edge_param_eid"].to_numpy().astype(np.int)
            inversemodel.edge_param_pm_range = df_parameter_space["edge_param_pm_range"].to_numpy().astype(np.double)

            if np.max(inversemodel.edge_param_eid) > flownetwork.nr_of_es - 1:
                sys.exit("Error: Edge parameter refers to invalid edge id.")

        inversemodel.nr_of_edge_parameters = np.size(inversemodel.edge_param_eid)
        inversemodel.nr_of_parameters = inversemodel.nr_of_edge_parameters