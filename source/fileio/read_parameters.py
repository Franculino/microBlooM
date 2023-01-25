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
            # The range is identical for all parameters (corresponds to the first value in the column).
            inversemodel.parameter_pm_range = np.ones(flownetwork.nr_of_es) * \
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
            inversemodel.parameter_pm_range = df_parameter_space["edge_param_pm_range"].to_numpy().astype(np.double)

            if np.max(inversemodel.edge_param_eid) > flownetwork.nr_of_es - 1:
                sys.exit("Error: Edge parameter refers to invalid edge id.")

        inversemodel.nr_of_edge_parameters = np.size(inversemodel.edge_param_eid)
        inversemodel.nr_of_parameters = inversemodel.nr_of_edge_parameters


class ReadParametersVertices(ReadParameters):
    """
    Class for importing the parameter space related to vertices (here, generally boundary conditions)
    """

    def read(self, inversemodel, flownetwork):
        """
        Import vertex ids of parameters and the corresponding allowable change (1 +- range)
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        import pandas as pd
        # Extract file path of target values.
        path_vertex_parameterspace = self._PARAMETERS["csv_path_vertex_parameterspace"]

        # Read file with pandas
        df_parameter_space = pd.read_csv(path_vertex_parameterspace)

        # Check for keyword all to include all edge ids into the parameter space.
        if df_parameter_space['vertex_param_vid'][0] == 'all':
            # All edge ids are parameters
            nr_of_boundaries = np.size(flownetwork.boundary_vs)
            inversemodel.vertex_param_vid = flownetwork.boundary_vs
            # The range is identical for all parameters (corresponds to the first value in the column).
            inversemodel.parameter_pm_range = np.ones(nr_of_boundaries) * \
                                              df_parameter_space['vertex_param_pm_range'][0]
        else:
            # Sort prescribed edge ids with parameters according to ascending edge ids.
            df_parameter_space = df_parameter_space.sort_values('vertex_param_vid')
            # Check for duplicate eids
            if True in df_parameter_space.duplicated(subset=['vertex_param_vid']).to_numpy():
                sys.exit("Error: Duplicate edge id in parameter space definition.")

            # Assign data to inversemodel object
            # Edge attributes
            inversemodel.vertex_param_vid = df_parameter_space["vertex_param_vid"].to_numpy().astype(np.int)
            inversemodel.parameter_pm_range = df_parameter_space["vertex_param_pm_range"].to_numpy().astype(np.double)

            is_a_boundary_vertex = np.in1d(inversemodel.vertex_param_vid, flownetwork.boundary_vs)
            if False in is_a_boundary_vertex:
                sys.exit("Error: Vertex parameter vid refers to vertex, which is not a boundary.")

        is_a_boundary_parameter = np.in1d(flownetwork.boundary_vs, inversemodel.vertex_param_vid)
        boundary_parameter_type = flownetwork.boundary_type[is_a_boundary_parameter]

        if True in (boundary_parameter_type > 1):  # Check if there are flow rate boundaries.
            sys.exit("Error: Currently, cannot tune flow rate boundary conditions. --> Use pressure boundaries.")

        inversemodel.nr_of_vertex_parameters = np.size(inversemodel.vertex_param_vid)
        inversemodel.nr_of_parameters = inversemodel.nr_of_vertex_parameters
