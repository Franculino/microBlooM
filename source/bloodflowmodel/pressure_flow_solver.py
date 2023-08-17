import sys
from abc import ABC, abstractmethod
from types import MappingProxyType

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from pyamg import smoothed_aggregation_solver


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
        nr_of_es = flownetwork.nr_of_es

        # Update flow rates based on the transmissibility and pressure.
        flow_rate = transmiss * (pressure[edge_list[:, 0]] - pressure[edge_list[:, 1]])

        # Update flow rate
        if flownetwork.zeroFlowThreshold is not None:
            # in case we want to exclude the unrealistic lower values in iterative model
            flownetwork.flow_rate = _update_low_flow(flownetwork, flow_rate)
        else:
            flownetwork.flow_rate = flow_rate


def set_low_flow_threshold(flownetwork, local_balance):
    # max of the mass balance error for the internal nodes
    flownetwork.zeroFlowThreshold = np.max(local_balance)

    # check how the flow it will change
    # Print to display the percentage of Zero flow vessel
    print(f"Percentage of zero flow vessel {np.round((len(flownetwork.flow_rate[flownetwork.flow_rate == 0]) / flownetwork.nr_of_es) * 100, decimals=2)}%")
    # Print to display the percentage of Zero flow vessel
    print(f"Min flow rate = {np.min(np.abs(flownetwork.flow_rate[flownetwork.flow_rate != 0]))} and max flow_rate = {np.max(np.abs(flownetwork.flow_rate))}")

    # print to check the value of the threshold
    print("Tolerance :" + str(flownetwork.zeroFlowThreshold))
    # update the flow rate
    return _update_low_flow(flownetwork, flownetwork.flow_rate)


def _update_low_flow(flownetwork, flow_rate):
    # Update flow rate based on the zero flow threshold
    flow_rate = np.where(np.abs(flow_rate) < flownetwork.zeroFlowThreshold, 0, flow_rate)

    # check how the flow it is changed
    # Print to display the percentage of Zero flow vessel
    print(f"Percentage of zero flow vessel {np.round((len(flow_rate[flow_rate == 0]) / flownetwork.nr_of_es) * 100, decimals=2)}")
    # Print to display the min and max flow_rate
    print(f"Min flow rate = {np.min(np.abs(flow_rate[flow_rate != 0]))} and max flow_rate = {np.max(np.abs(flow_rate))} %")

    return flow_rate


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
