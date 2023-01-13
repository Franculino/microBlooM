from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class ReadTargetValues(ABC):
    """
    Abstract base class for the implementations related to importing the target values of the inverse problem
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of ReadTargetValues.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def read(self, inversemodel):
        """
        Read target values
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """


class ReadTargetValuesEdge(ReadTargetValues):
    """
    Class for importing the target values related to edges
    """

    def read(self, inversemodel):
        """
        Import target values and types for edges
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        """
        import pandas as pd
        # Extract file path of target values.
        path_edge_target_data = self._PARAMETERS["csv_path_edge_target_data"]

        # Read files with pandas, sort and check for duplicates
        df_target_data = pd.read_csv(path_edge_target_data)

        df_target_data = df_target_data.sort_values('edge_tar_eid')  # sort according to ascending eids

        if True in df_target_data.duplicated(subset=['edge_tar_eid']).to_numpy():  # check for duplicate eids
            sys.exit("Error: Duplicate edge id in constraint definition.")

        # Assign data to inversemodel class
        # Edge attributes
        inversemodel.edge_constraint_eid = df_target_data["edge_tar_eid"].to_numpy().astype(np.int)
        inversemodel.edge_constraint_type = df_target_data["edge_tar_type"].to_numpy().astype(np.int)
        inversemodel.edge_constraint_value = df_target_data["edge_tar_value"].to_numpy().astype(np.double)
        inversemodel.edge_constraint_range_pm = df_target_data["edge_tar_range_pm"].to_numpy().astype(np.double)
        inversemodel.edge_constraint_sigma = df_target_data["edge_tar_sigma"].to_numpy().astype(np.double)

        inversemodel.nr_of_edge_constraints = np.size(inversemodel.edge_constraint_eid)