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
    def read(self, inversemodel, flownetwork):
        """
        Read target values
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class ReadTargetValuesEdge(ReadTargetValues):
    """
    Class for importing the target values related to edges
    """

    def read(self, inversemodel, flownetwork):
        """
        Import target values and types for edges
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        import pandas as pd
        # Extract file path of target values.
        path_edge_target_data = self._PARAMETERS["csv_path_edge_target_data"]

        # Read files with pandas, sort and check for duplicates
        df_target_data = pd.read_csv(path_edge_target_data)

        df_measurement_data = pd.read_csv(self._PARAMETERS["csv_path_edge_target_measurements"])
        df_measurement_data = df_measurement_data.sort_values('edge_tar_eid')

        df_target_data = df_target_data.sort_values('edge_tar_eid')  # sort according to ascending eids

        if True in df_target_data.duplicated(subset=['edge_tar_eid']).to_numpy():  # check for duplicate eids
            sys.exit("Error: Duplicate edge id in constraint definition.")

        # Assign data to inversemodel class
        # Edge attributes
        inversemodel.edge_constraint_eid = df_target_data["edge_tar_eid"].to_numpy().astype(int)
        inversemodel.edge_constraint_type = df_target_data["edge_tar_type"].to_numpy().astype(int)
        inversemodel.edge_constraint_value = df_target_data["edge_tar_value"].to_numpy().astype(np.double)
        inversemodel.edge_constraint_range_pm = df_target_data["edge_tar_range_pm"].to_numpy().astype(np.double)
        inversemodel.edge_constraint_sigma = df_target_data["edge_tar_sigma"].to_numpy().astype(np.double)
        inversemodel.measurements_value = df_measurement_data["edge_tar_value"].to_numpy().astype(np.double)
        inversemodel.measurements_eid = df_measurement_data["edge_tar_eid"].to_numpy().astype(int)

        # Vertices of each measurement or range
        edge_numbers = flownetwork.edge_list[inversemodel.measurements_eid]
        inversemodel.targets_boundary_v = np.intersect1d(edge_numbers, flownetwork.boundary_vs)
        ranges_vertices = flownetwork.edge_list[inversemodel.edge_constraint_eid]
        inversemodel.ranges_boundary_v = np.setdiff1d(flownetwork.boundary_vs, inversemodel.targets_boundary_v)
        # Crear una matriz booleana que indique si cada vértice está en flownetwork.boundary_vs
        is_in_boundary = np.isin(edge_numbers, flownetwork.boundary_vs)
        is_in_boundary_range = np.isin(ranges_vertices, flownetwork.boundary_vs)

        # Verificar si al menos uno de los vértices está en flownetwork.boundary_vs para cada par de vértices
        any_in_boundary = np.any(is_in_boundary, axis=1)
        any_in_boundary_range = np.any(is_in_boundary_range, axis=1)

        # Filtrar los inversemodel.measurements_eid correspondientes a los pares de vértices que cumplen con la condición
        inversemodel.targets_boundary_ed = inversemodel.measurements_eid[any_in_boundary]
        inversemodel.ranges_boundary_ed = np.setdiff1d(inversemodel.edge_constraint_eid[any_in_boundary_range]
                                                      , inversemodel.targets_boundary_ed)

        inversemodel.nr_of_edge_constraints = np.size(inversemodel.edge_constraint_eid)

        if np.max(inversemodel.edge_constraint_eid) > flownetwork.nr_of_es - 1:
            sys.exit("Error: Edge constraint refers to invalid edge id.")
