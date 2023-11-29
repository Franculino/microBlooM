import sys
from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
#from line_profiler_pycharm import profile
from scipy.sparse import coo_matrix, csc_matrix
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


class BuildSystemSparseCoo(BuildSystem):
    """
    Class for building a sparse linear system of equations (coo_matrix).
    """

    #@profile
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
        flownetwork.system_matrix = coo_matrix((data, (row, col)), shape=(nr_of_vs, nr_of_vs)) #Modifica TODO
        flownetwork.rhs = rhs

class BuildSystemSparseCsc(BuildSystem):
    """
    Class for building a sparse linear system of equations (csc_matrix).
    """

    #@profile
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


class BuildSystemSparseCooNoOneSimple(BuildSystem):
    """
    Class for building a sparse linear system of equations (coo_matrix) without one in the matrix.
    Valid in case of pressure
    """

    #@profile
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
        for vid, value, types in zip(boundary_vertices, boundary_values, boundary_types):
            if types == 1:  # If is a pressure boundary
                data[row == vid] = 0.

        row = np.append(row, boundary_vertices[boundary_types == 1])
        col = np.append(col, boundary_vertices[boundary_types == 1])
        # Add 1/boundary values to diagonal entries of system matrix that represent pressure boundary vertices.
        fractional_boundary = 1 / boundary_values[boundary_types == 1]
        data = np.append(data, fractional_boundary)

        # Assign 1 to the pressure boundary vertices to right hand side vector.
        rhs[boundary_vertices[boundary_types == 1]] = 1

        # Assign flow rate boundary value to right hand side vector.
        rhs[boundary_vertices[boundary_types == 2]] = boundary_values[boundary_types == 2]  # assign flow source term to rhs

        # Build the system matrix and assign the right hand side vector.
        system_matrix = coo_matrix((data, (row, col)), shape=(nr_of_vs, nr_of_vs))
        flownetwork.system_matrix = system_matrix
        flownetwork.rhs = rhs

        pressure = spsolve(flownetwork.system_matrix, flownetwork.rhs)
        residual = (flownetwork.system_matrix * pressure) - rhs

        # Compute the residual of the internal nodes
        # boolean mask to identify the element to be eliminated
        mask = np.ones(residual.shape, dtype=bool)
        mask[boundary_vertices] = False

        # Delete elements at boundary vertex
        residual = residual[mask]
        # residuals
        flownetwork.residualsInternalNodesSimple = residual


class BuildSystemSparseCooNoOne(BuildSystem):
    """
    Class for building a sparse linear system of equations (coo_matrix) without one in the matrix.
    """

    #@profile
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

        # Account for boundary conditions.
        boundary_vertices = flownetwork.boundary_vs
        boundary_values = flownetwork.boundary_val
        boundary_types = flownetwork.boundary_type  # 1: pressure, 2: flow rate

        # Pressure boundaries.
        # Set entire rows of the system matrix that represent pressure boundary vertices to 0.
        for vid, value, types in zip(boundary_vertices, boundary_values, boundary_types):
            if types == 1:  # If is a pressure boundary
                data[row == vid] = 0.

        # Add 1 to diagonal entries of system matrix that represent pressure boundary vertices.
        row = np.append(row, boundary_vertices[boundary_types == 1])
        col = np.append(col, boundary_vertices[boundary_types == 1])
        data = np.append(data, np.ones(np.size(boundary_vertices[boundary_types == 1])))

        # The method to create the system change from here

        # Build the system matrix (using the CSC format)
        system_matrix = csc_matrix((data, (row, col)), shape=(nr_of_vs, nr_of_vs))

        # Initialise the right hand side vector.
        rhs = np.zeros(nr_of_vs)

        # Assign flow rate boundary value to right hand side vector.
        rhs[boundary_vertices[boundary_types == 2]] = boundary_values[boundary_types == 2]  # assign flow source term to rhs

        # Pressure values
        # index of column
        for column_index in boundary_vertices:
            # Identify the column in the system matrix
            column_vector = system_matrix.getcol(column_index)
            # Find all the element present in the system matrix
            elements = column_vector.data
            # Identify the rows of the system matrix
            rows = column_vector.indices

            aux = np.zeros(flownetwork.nr_of_vs)
            aux[flownetwork.boundary_vs[flownetwork.boundary_type == 1]] = flownetwork.boundary_val[flownetwork.boundary_type == 1]

            # reconstruct the pressure arrays
            flownetwork.pressure = aux
            # Identify the element that need to go on the right side vector
            for element, row_idx in zip(elements, rows):
                # remove the ones from the possible values in the data
                element = np.where(element == 1, 0, element)
                # sum the element for that row
                rhs[row_idx] += np.abs(element) * aux[row_idx]

        # Modify system matrix and the rhs to eliminate the rows and the column of the boundary nodes
        # System

        # cast to CSR format
        system_matrix = system_matrix.tocsr()

        # Create boolean mask, to select the one to be eliminated
        keep_mask = np.ones(system_matrix.shape[0], dtype=bool)
        keep_mask[boundary_vertices] = False
        sliced_system_matrix = system_matrix[keep_mask][:, keep_mask]

        # cast back to CSC
        system_matrix = sliced_system_matrix.tocsc()

        # RHS
        # boolean mask to identify the element to be eliminated
        mask = np.ones(rhs.shape, dtype=bool)
        mask[boundary_vertices] = False

        # Delete elements at boundary vertex
        rhs = rhs[mask]

        # Assign the values
        flownetwork.system_matrix = system_matrix
        flownetwork.rhs = rhs

        # to check residuals
        pressure = spsolve(flownetwork.system_matrix, flownetwork.rhs)

        # Compute the residual
        flownetwork.residualsInternalNodesOne = (system_matrix * pressure) - rhs
