import copy
import sys
from abc import ABC, abstractmethod
from types import MappingProxyType

import numpy as np
from line_profiler_pycharm import profile
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, norm, inv

import scipy.sparse as sparse
from pyamg import smoothed_aggregation_solver
from scipy.sparse import csr_matrix
from scipy.sparse import isspmatrix_csc

from source.bloodflowmodel.build_system import BuildSystemSparseCooNoOne, BuildSystemSparseCooNoOneSimple


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

    @profile
    def _solve_flow(self, flownetwork):
        """
        Solve for the flow rates and update the flow rates in flownetwork. Note that negative flow rates correspond
        to a flow in negative edge direction.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        edge_list = flownetwork.edge_list
        transmiss = flownetwork.transmiss
        pressure = flownetwork.pressure

        #
        if flownetwork.tollerance is None:
            # Update flow rates based on the transmissibility and pressure.
            flownetwork.flow_rate = transmiss * (pressure[edge_list[:, 0]] - pressure[edge_list[:, 1]])
            print("Number of zero flow vessel " + str(len(flownetwork.flow_rate[flownetwork.flow_rate == 0])) + " "
                  + str(np.round((len(flownetwork.flow_rate[flownetwork.flow_rate == 0]) / flownetwork.nr_of_es) * 100, decimals=2)) + "%")
        else:
            flow_rate = transmiss * (pressure[edge_list[:, 0]] - pressure[edge_list[:, 1]])
            flownetwork.flow_rate = np.where(np.abs(flow_rate) < flownetwork.tollerance, 0, flow_rate)
            print("Number of zero flow vessel " + str(len(flownetwork.flow_rate[flownetwork.flow_rate == 0])) + " "
                  + str(np.round((len(flownetwork.flow_rate[flownetwork.flow_rate == 0]) / flownetwork.nr_of_es) * 100, decimals=2)) + "%")

        # flow_rate = transmiss * (pressure[edge_list[:, 0]] - pressure[edge_list[:, 1]])
        # flownetwork.flow_rate = np.where(np.abs(flow_rate) < self._PARAMETERS["machine_error"], 0, flow_rate)

        # :TODO: REFACTOR AND FINALIZE THE IMPLEMENTATION
        # system_matrix = flownetwork.system_matrix

        #
        # # cast the matrix CSC to identify max and min
        # system_matrix = system_matrix.tocsc()
        # print("Min value of system matrix " + str(np.abs(system_matrix.data[system_matrix.data != 0]).min()))
        # print("Max value of system matrix " + str(np.abs(system_matrix.data[system_matrix.data != 1]).max()))
        #
        # # Calculate the 2-norm of the inverse of the CSR matrix
        # norm_csr = norm(system_matrix)
        # # Calculate the 2-norm of the inverse of the CSR matrix
        # norm_inv_csr = norm(inv(system_matrix))
        # # Calculate the condition number
        # cond = norm_csr * norm_inv_csr
        # # Calculate number of accurate digits
        # significant_digits = np.round((16 - np.log10(cond)), decimals=0)
        # print("Condition number" + str(cond))
        # print("Assuming as A and b accurate up to 16 decimal digits, the entries are accurate of 16 - "
        # + str(np.round(np.log10(cond))) + " = " + str(significant_digits) + " digit")
        #
        # # To have all the number in the significant_digits
        # print(pressure[13703])
        # arr_pressure = np.array([format(x, f".{int(significant_digits)}") for x in pressure], dtype=float)
        # print(arr_pressure[13703])
        # pressure_diff = arr_pressure[edge_list[:, 0]] - arr_pressure[edge_list[:, 1]]
        #
        # print("Min pressure difference" + str(np.min(np.abs(pressure_diff[pressure_diff != 0]))))
        # print("Max pressure difference" + str(np.max(np.abs(pressure_diff))))
        # sys.exit()
        #
        # # pressure_diff_eps = np.where(np.abs(pressure_diff) < flownetwork.eps_eff, 0, pressure_diff)
        #
        # flownetwork.flow_rate = transmiss * pressure
        #
        # # print("flow rate: max and min")
        # print("Total number of vessel " + str(flownetwork.nr_of_es))
        # print("Number of zero flow vessel " + str(len(flownetwork.flow_rate[flownetwork.flow_rate == 0])))
        # print("Percentage of zero flow vessel " + str((len(flownetwork.flow_rate[flownetwork.flow_rate == 0]) / flownetwork.nr_of_es) * 100))
        #
        # # flownetwork.min_flow = np.min(abs(flownetwork.flow_rate[flownetwork.flow_rate != 0]))


class PressureFlowSolverSparseDirect(PressureFlowSolver):
    """
    Class for calculating the pressure with a sparse direct solver.
    """

    @profile
    def _solve_pressure(self, flownetwork):
        """
        Solve the linear system of equations for the pressure and update the pressure in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.pressure = spsolve(csc_matrix(flownetwork.system_matrix), flownetwork.rhs)

        # new approach
        # BuildSystemSparseCooNoOneSimple.build_linear_system(self, flownetwork)
        # print(np.sum(np.absolute(flownetwork.residualsInternalNodesOne)) / (flownetwork.nr_of_vs - len(flownetwork.boundary_vs)))
        # print(np.sum(np.absolute(flownetwork.residualsInternalNodesSimple)) / (flownetwork.nr_of_vs - len(flownetwork.boundary_vs)))
        #
        # sys.exit()
        # if isspmatrix_csc(flownetwork.system_matrix):
        #
        #     # Compute the pressures
        #     pressure = spsolve(flownetwork.system_matrix, flownetwork.rhs)
        #
        #     # Compute the residual
        #     residual = (flownetwork.system_matrix * pressure) - flownetwork.rhs
        #
        #     # Create a new array (aux) and insert elements from boundary values in it at specific position
        #     aux = np.zeros(flownetwork.nr_of_vs)
        #     aux[flownetwork.boundary_vs[flownetwork.boundary_type == 1]] = flownetwork.boundary_val[flownetwork.boundary_type == 1]
        #
        #     # Insert the pressure in the auxiliar array in the remaining spot
        #     aux[~np.isin(np.arange(len(aux)), flownetwork.boundary_vs[flownetwork.boundary_type == 1])] = pressure
        #
        #     # reconstruct the pressure arrays
        #     flownetwork.pressure = aux
        # else:
        #     flownetwork.pressure = spsolve(csc_matrix(flownetwork.system_matrix), flownetwork.rhs)
        #     # Compute the residual
        #     residual = (flownetwork.system_matrix * flownetwork.pressure) - flownetwork.rhs


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
