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
    def read(self, inversemodel):
        """
        Read parameter space
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """


class ReadParametersEdges(ReadParameters):
    """
    Class for importing the parameter space related to edges
    """

    def read(self, inversemodel):
        """
        Import target values and types for edges
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """
        import pandas as pd
        # Extract file path of target values.
        path_edge_parameterspace = self._PARAMETERS["csv_path_edge_parameterspace"]

        # Read files with pandas, sort and check for duplicates
        df_parameter_space = pd.read_csv(path_edge_parameterspace)

        df_parameter_space = df_parameter_space.sort_values('edge_param_eid')  # sort according to ascending eids
        if True in df_parameter_space.duplicated(subset=['edge_param_eid']).to_numpy():  # check for duplicate eids
            sys.exit("Error: Duplicate edge id in parameter space definition.")

        # Assign data to inversemodel class
        # Edge attributes
        inversemodel.edge_param_eid = df_parameter_space["edge_param_eid"].to_numpy().astype(np.int)
        inversemodel.edge_param_pm_range = df_parameter_space["edge_param_pm_range"].to_numpy().astype(np.double)

        inversemodel.nr_of_edge_parameters = np.size(inversemodel.edge_param_eid)