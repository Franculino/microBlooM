import copy
import sys
from abc import ABC, abstractmethod
from types import MappingProxyType

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
from scipy.sparse.linalg import spsolve, norm, inv
from pyamg import smoothed_aggregation_solver

from source.bloodflowmodel.build_system import BuildSystemSparseCscNoOne, BuildSystemSparseCooNoOneSimple


class PressureFlowSolver(ABC):
    """
    Abstract base class for the implementations related to the linear solver for the blood flow model.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of PressureFlowSolver.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def _solve_pressure(self, flownetwork):
        """
        Solve the linear system of equations for the pressure and update the pressure in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    def update_pressure_flow(self, flownetwork):
        """
        Call the functions that solve for the pressures and flow rates.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        self._solve_pressure(flownetwork)
        self._solve_flow(flownetwork)

    def _solve_flow(self, flownetwork):
        """
        Solve for the flow rates and update the flow rates in flownetwork. Note that negative flow rates correspond
        to a flow in negative edge direction.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        edge_list = flownetwork.edge_list
        transmiss = flownetwork.transmiss
        system_matrix = flownetwork.system_matrix
        pressure_One = flownetwork.pressure_One
        pressure_Original = flownetwork.pressure_Original
        pressure_OneSimple = flownetwork.pressure_OneSimple

        # Update flow rates based on the transmissibility and pressure.
        flownetwork.flow_rateOne = transmiss * (pressure_One[edge_list[:, 0]] - pressure_One[edge_list[:, 1]])
        flownetwork.flow_rateOneSimple = transmiss * (pressure_OneSimple[edge_list[:, 0]] - pressure_OneSimple[edge_list[:, 1]])
        flownetwork.flow_rateOriginal = transmiss * (pressure_Original[edge_list[:, 0]] - pressure_Original[edge_list[:, 1]])

        # # Calculate the 2-norm of the inverse of the CSR matrix
        # norm_csr = norm(system_matrix)
        # # Calculate the 2-norm of the inverse of the CSR matrix
        # norm_inv_csr = norm(inv(system_matrix))
        # # Calculate the condition number
        # cond = norm_csr * norm_inv_csr
        # # Calculate number of accurate digits
        # significant_digits = np.round((16 - np.log10(cond)), decimals=0)
        # print("Condition number" + str(cond))
        # print("Assuming as A and b accurate up to 16 decimal digits, the entries are accurate of 16 - " + str(np.round(np.log10(cond))) + " = " +
        #   str(significant_digits) + " digit")

        # sys.exit()


class PressureFlowSolverSparseDirect(PressureFlowSolver):
    """
    Class for calculating the pressure with a sparse direct solver.
    """

    def _solve_pressure(self, flownetwork):
        """
        Solve the linear system of equations for the pressure and update the pressure in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.pressure = spsolve(csc_matrix(flownetwork.system_matrix), flownetwork.rhs)


class PressureFlowSolverSparseDirectImprove(PressureFlowSolver):
    """
    Class for calculating the pressure with a sparse direct solver in case of:
    - CSC matrix: this matrix is created from the BuildSystemSparseCscNoOne(BuildSystem), the entries refer just to the internal nodes
    - other matrix: this matrix is created from any other methods, it is referred to all the nodes

    """

    def _solve_pressure(self, flownetwork):
        """
        Solve the linear system of equations for the pressure and update the pressure in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        system_matrix = flownetwork.system_matrix
        rhs = flownetwork.rhs
        boundary_vs = flownetwork.boundary_vs
        boundary_val = flownetwork.boundary_val
        boundary_type = flownetwork.boundary_type
        nr_of_vs = flownetwork.nr_of_vs

        if isspmatrix_csc(system_matrix):
            # Compute the pressures for internal nodes
            pressure_internal_nodes = spsolve(system_matrix, rhs)

            # Compute the residual
            residual_Internal_Node = flownetwork.system_matrix.dot(pressure_internal_nodes) - flownetwork.rhs

            # Create a new array (aux)
            aux = np.zeros(nr_of_vs)

            # Insert the boundary values in the position in aux vertex equal to boundary vertex
            aux[boundary_vs[boundary_type == 1]] = boundary_val[boundary_type == 1]
            aux[boundary_vs[boundary_type == 2]] = boundary_val[boundary_type == 2]

            # Insert the pressure in the aux array in the remaining spot
            aux[~np.isin(np.arange(len(aux)), boundary_vs[(boundary_type == 1) | (boundary_type == 2)])] = pressure_internal_nodes

            # Reconstruct the pressure arrays
            flownetwork.pressure = aux

        else:
            flownetwork.pressure = spsolve(csc_matrix(system_matrix), rhs)

            # Compute the residual
            residualFullNodes = system_matrix.dot(flownetwork.pressure) - rhs
        sys.exit()


class PressureFlowSolverSparseDirectImproveUtil(PressureFlowSolver):
    """
    UTIL METHOD TO PRINT THE RESULT [TO BE DELETED]

    Class for calculating the pressure with a sparse direct solver in case of:
    - CSC matrix: this matrix is created from the BuildSystemSparseCscNoOne(BuildSystem), the entries refer just to the internal nodes
    - other matrix: this matrix is created from any other methods, it is referred to all the nodes
    """

    def _solve_pressure(self, flownetwork):
        """
        Solve the linear system of equations for the pressure and update the pressure in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.pressure_Original = spsolve(csc_matrix(flownetwork.system_matrix), flownetwork.rhs)
        pressure_Original = copy.copy(flownetwork.pressure_Original)
        # Compute the residual
        residual_Old_method_Full = flownetwork.system_matrix.dot(flownetwork.pressure_Original) - flownetwork.rhs

        # SIMPLE ONE APPROACH
        BuildSystemSparseCooNoOneSimple.build_linear_system(self, flownetwork)
        flownetwork.pressure_OneSimple = spsolve(csc_matrix(flownetwork.system_matrix), flownetwork.rhs)
        pressure_OneSimple = copy.copy(flownetwork.pressure_OneSimple)
        # Compute the residual
        residual_OneSimple_Full = flownetwork.system_matrix.dot(flownetwork.pressure_OneSimple) - flownetwork.rhs

        # NEW SYSTEM CREATION
        BuildSystemSparseCscNoOne.build_linear_system(self, flownetwork)  # is csc
        if isspmatrix_csc(flownetwork.system_matrix):
            # Compute the pressures
            pressure = spsolve(flownetwork.system_matrix, flownetwork.rhs)

            # Compute the residual
            residual_Internal_Node = flownetwork.system_matrix.dot(pressure) - flownetwork.rhs

            # Create a new array (aux) and insert elements from boundary values in it at specific position
            aux = np.zeros(flownetwork.nr_of_vs)
            aux[flownetwork.boundary_vs[flownetwork.boundary_type == 1]] = flownetwork.boundary_val[flownetwork.boundary_type == 1]
            aux[flownetwork.boundary_vs[flownetwork.boundary_type == 2]] = flownetwork.boundary_val[flownetwork.boundary_type == 2]

            # Insert the pressure in the auxiliar array in the remaining spot
            aux[~np.isin(np.arange(len(aux)), flownetwork.boundary_vs[(flownetwork.boundary_type == 1) | (flownetwork.boundary_type == 2)])] = pressure

            # reconstruct the pressure arrays
            flownetwork.pressure_One = aux
            pressure_One = flownetwork.pressure_One
        else:
            flownetwork.pressure = spsolve(csc_matrix(flownetwork.system_matrix), flownetwork.rhs)
            # Compute the residual
            residual = flownetwork.system_matrix.dot(flownetwork.pressure) - flownetwork.rhs

        # Pressure
        print("Difference between pressure computed with original method and new simple approach")
        are_arrays_similar(pressure_Original, pressure_OneSimple, 1)
        print("Difference between pressure computed with original method and  new approach")
        are_arrays_similar(pressure_Original, pressure_One, 1)

        # RESIDUALS
        print("Max residual computed with original method (FULL) " + str(np.max(residual_Old_method_Full)))
        print("Max residual computed with new simple approach (FULL) " + str(np.max(residual_OneSimple_Full)))

        print("Max residual computed with original method (internal) " + str(np.max(internal_nodes(residual_Old_method_Full, flownetwork.boundary_vs))))
        print("Max residual computed with new simple approach (internal) " + str(np.max(internal_nodes(residual_OneSimple_Full, flownetwork.boundary_vs))))
        print("Max residual computed with Noone approach (internal) " + str(np.max(residual_Internal_Node)))

        # sys.exit()


def internal_nodes(pressure, boundary_vertices):
    # RHS
    # boolean mask to identify the element to be eliminated
    mask = np.ones(pressure.shape, dtype=bool)
    mask[boundary_vertices] = False

    # Delete elements at boundary vertex
    pressure = pressure[mask]
    return pressure


def are_arrays_similar(arr1, arr2, threshold_percent):
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    threshold = threshold_percent / 100.0 * np.max(np.abs(arr1))
    percentage_diff = (max_diff / np.max(np.abs(arr1))) * 100.0
    if max_diff <= threshold:
        return print("Percentage of difference " + str(percentage_diff) + " %" + " Max difference " + str(max_diff))
    else:
        return print("Percentage of difference " + str(percentage_diff) + " %" + " Max difference " + str(max_diff))


class PressureFlowSolverPyAMG(PressureFlowSolver):
    """
    Class for calculating the pressure with an algebraic multigrid (AMG) solver.
    """

    def _solve_pressure(self, flownetwork, tol=1.00E-14):
        """
        Solve the linear system of equations for the pressure and update the pressure in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        A = csr_matrix(flownetwork.system_matrix)
        b = flownetwork.rhs
        # Create solver
        ml = smoothed_aggregation_solver(A, strength=('symmetric', {'theta': 0.0}),
                                         smooth=('energy', {'krylov': 'cg', 'maxiter': 2, 'degree': 1, 'weighting': 'local'}),
                                         improve_candidates=[
                                             ('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None,
                                             None, None, None, None, None, None, None, None, None, None, None, None,
                                             None],
                                         aggregate="standard",
                                         presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                                         postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
                                         max_levels=15,
                                         max_coarse=500,
                                         coarse_solver="pinv")
        # Solve system
        tol_solver = np.abs(np.min(flownetwork.system_matrix)) * tol  # tolerance

        # x0 is the initial guess for the pressure field.
        # For the blood flow model and the first iteration of the inverse model, we have to initialise the x0 array
        # based on boundary conditions. Otherwise, in case of the inverse model, x0 should be the previous solution.
        # Initialisation: In case of only pressure boundary conditions, x0 should be an array with values between
        # inlet and outlet pressure values. In case of pressure and flow rate boundary conditions, x0 should be an
        # array with values +/-50% of the pressure boundary value.
        res = []
        if flownetwork.pressure is None:
            if (1 in flownetwork.boundary_type) and not (2 in flownetwork.boundary_type):  # only pressure boundaries
                boundary_inlet = np.max(flownetwork.boundary_val)
                boundary_outlet = np.min(flownetwork.boundary_val)
                x0 = boundary_inlet - np.arange(0.001, 1, 0.999 / flownetwork.nr_of_vs) * (boundary_inlet - boundary_outlet)
                x0[flownetwork.boundary_vs] = flownetwork.boundary_val
            elif (1 in flownetwork.boundary_type) and (2 in flownetwork.boundary_type):
                boundary_pressure_vs = flownetwork.boundary_vs[flownetwork.boundary_type == 1]
                boundary_pressure_val = flownetwork.boundary_val[flownetwork.boundary_type == 1]
                x0 = np.arange(0.5, 1.5, 1 / flownetwork.nr_of_vs) * np.max(boundary_pressure_val)
                x0[boundary_pressure_vs] = boundary_pressure_val
            else:
                sys.exit("Warning Message: only flow boundary conditions were assigned! Define new boundary conditions,"
                         " including at least one pressure boundary condition!")
            flownetwork.pressure, info = ml.solve(b, x0=x0, tol=tol_solver, residuals=res, accel="cg", maxiter=600,
                                                  cycle="V", return_info=True)
        else:
            x0 = flownetwork.pressure
            flownetwork.pressure, info = ml.solve(b, x0=x0, tol=tol_solver, residuals=res, accel="cg", maxiter=600,
                                                  cycle="V", return_info=True)
        # Provides convergence information
        if not info == 0:  # if info is zero, successful exit from the iterative solver
            print("ERROR in Solving the Matrix")
