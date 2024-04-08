from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class BuildSystem(ABC):
    """
    Abstract base class for the implementations related to building the linear system of equations
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of BuildSystem.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def build_linear_system(self, flownetwork):
        """
        Build a linear system of equations for the pressure and update the system matrix and right hand side in
        flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class BuildSystemSparseCsc(BuildSystem):
    """
    Class for building a sparse linear system of equations (csc_matrix).
    """

    def build_linear_system(self, flownetwork):
        """
        Fast method to build a linear system of equation. The sparse system matrix is COOrdinate format. The right
        hand side vector is a 1d-numpy array. Accounts for pressure and flow boundary conditions. The system matrix
        and right hand side are updated in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        nr_of_vs = flownetwork.nr_of_vs
        transmiss = flownetwork.transmiss
        edge_list = flownetwork.edge_list
        # Generate row, col and data arrays required to build a coo_matrix.
        # In a first step, assume a symmetrical system matrix without accounting for boundary conditions.
        # Example: Network with 2 edges and 3 vertices. edge_list = [[v1, v2], [v2, v3]], transmiss = [T1, T2]
        # row = [v1, v2, v1, v2, v2, v3, v2, v3]
        # col = [v2, v1, v1, v2, v3, v2, v2, v3]
        # data = [-T1, -T1, T1, T1, -T2, -T2, T2, T2]
        row = np.concatenate((edge_list, edge_list), axis=1).reshape(-1)
        col = np.concatenate((np.roll(edge_list, 1, axis=1), edge_list), axis=1).reshape(-1)
        data = np.vstack([transmiss] * 4).transpose()  # Assign transmissibilities on diagonals and off-diagonals.
        data[:, :2] *= -1  # set off-diagonals to negative values.
        data = data.reshape(-1)

        # Initialise the right hand side vector.
        rhs = np.zeros(nr_of_vs)

        # Account for boundary conditions.
        boundary_vertices = flownetwork.boundary_vs
        boundary_values = flownetwork.boundary_val
        boundary_types = flownetwork.boundary_type  # 1: pressure, 2: flow rate

        # Pressure boundaries.
        # Set entire rows of the system matrix that represent pressure boundary vertices to 0.
        for vid, value, type in zip(boundary_vertices, boundary_values, boundary_types):
            if type == 1:  # If is a pressure boundary
                data[row == vid] = 0.
        # Add 1 to diagonal entries of system matrix that represent pressure boundary vertices.
        row = np.append(row, boundary_vertices[boundary_types == 1])
        col = np.append(col, boundary_vertices[boundary_types == 1])
        data = np.append(data, np.ones(np.size(boundary_vertices[boundary_types == 1])))

        # Assign pressure boundary value to right hand side vector.
        rhs[boundary_vertices[boundary_types == 1]] = boundary_values[boundary_types == 1]
        # Assign flow rate boundary value to right hand side vector.
        rhs[boundary_vertices[boundary_types == 2]] = boundary_values[boundary_types == 2]  # assign flow source term to rhs
        # Build the system matrix and assign the right hand side vector.
        flownetwork.system_matrix = csc_matrix((data, (row, col)), shape=(nr_of_vs, nr_of_vs))
        flownetwork.rhs = rhs


