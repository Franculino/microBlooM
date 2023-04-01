import sys
from abc import ABC, abstractmethod
from types import MappingProxyType

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, cg
from pyamg import smoothed_aggregation_solver, rootnode_solver

from source.solver_diagnostics import solver_diagnostics


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
        pressure = flownetwork.pressure

        # Update flow rates based on the transmissibility and pressure.
        flownetwork.flow_rate = transmiss * (pressure[edge_list[:, 0]] - pressure[edge_list[:, 1]])


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
        A = csr_matrix(flownetwork.system_matrix); b = flownetwork.rhs
        B = np.ones((A.shape[0], 1), dtype=A.dtype); BH = B.copy()
        # Create solver
        ml = smoothed_aggregation_solver(A, B=B, BH=BH,
                                         strength=('symmetric', {'theta': 0.0}),
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

        ##
        # Solve system
        tol_solver = np.abs(np.min(flownetwork.system_matrix)) * tol
        res = []
        if flownetwork.pressure is None:
            if (1 in flownetwork.boundary_type) and not(2 in flownetwork.boundary_type):  # if there's pressure boundary
                boundary_inlet = np.max(flownetwork.boundary_val)
                boundary_outlet = np.min(flownetwork.boundary_val)
                x0 = boundary_inlet - np.arange(0.001, 1, 0.999/flownetwork.nr_of_vs) * (boundary_inlet - boundary_outlet)
                x0[flownetwork.boundary_vs] = flownetwork.boundary_val
            elif (1 in flownetwork.boundary_type) and (2 in flownetwork.boundary_type):
                boundary_pressure_vs = flownetwork.boundary_vs[flownetwork.boundary_type==1]
                boundary_pressure_val = flownetwork.boundary_val[flownetwork.boundary_type==1]
                x0 = np.arange(0.001, 1, 0.999/flownetwork.nr_of_vs) * np.max(boundary_pressure_val)
                x0[boundary_pressure_vs] = boundary_pressure_val
            else:
                sys.exit("Warning Message: only flow boundary conditions were assigned! Define new boundary conditions,"
                         " including at least one pressure boundary condition!")
            flownetwork.pressure= ml.solve(b, x0=x0, tol=tol_solver, residuals=res, accel="cg", maxiter=600, cycle="V")
        else:
            x0 = flownetwork.pressure
            flownetwork.pressure = ml.solve(b, x0=x0, tol=tol_solver, accel="cg", maxiter=600, cycle="V")

        # solver_diagnostics.solver_diagnostics(csr_matrix(flownetwork.system_matrix), fname='iso_diff_diagnostic', cycle_list=['V'])
        # sys.exit()

        # ml = rootnode_solver(csr_matrix(flownetwork.system_matrix), smooth='energy')
        # resvec = []
        # flownetwork.pressure = ml.solve(flownetwork.rhs, tol=1e-50, residuals=resvec)

        # for i, r in enumerate(resvec):
        #     print("residual at iteration {0:2}: {1:^6.2e}".format(i, r))

        # ml = smoothed_aggregation_solver(csr_matrix(flownetwork.system_matrix))  # AMG solver
        # M = ml.aspreconditioner(cycle='V')  # preconditioner
        # tol_solver = np.abs(np.min(flownetwork.system_matrix)) * tol
        #
        # if flownetwork.pressure is None:
        #     boundary_inlet = np.max(flownetwork.boundary_val)
        #     boundary_outlet = np.min(flownetwork.boundary_val)
        #     random = np.random.rand(flownetwork.system_matrix.shape[0])
        #     x0 = random * (boundary_inlet - boundary_outlet) + boundary_outlet
        #     flownetwork.pressure, _ = cg(flownetwork.system_matrix, flownetwork.rhs, x0=x0, tol=tol_solver, M=M)  # solve with CG
        # else:
        #     flownetwork.pressure, _ = cg(flownetwork.system_matrix, flownetwork.rhs, x0=flownetwork.pressure,
        #                                  tol=tol_solver*tol, M=M)  # solve with CG
                                         

