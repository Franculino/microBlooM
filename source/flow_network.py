import numpy as np

import source.fileio.read_network as readnetwork
import source.fileio.write_network as writenetwork
import source.bloodflowmodel.tube_haematocrit as tubehaematocrit
import source.bloodflowmodel.discharge_haematocrit as dischargehaematocrit
import source.bloodflowmodel.transmissibility as transmissibility
import source.bloodflowmodel.pressure_flow_solver as pressureflowsolver
import source.bloodflowmodel.build_system as buildsystem
import source.bloodflowmodel.rbc_velocity as rbc_velocity
import source.bloodflowmodel.iterative as iterative_routine
import source.bloodflowmodel.flow_balance as flow_balance
from types import MappingProxyType


class FlowNetwork(object):
    # todo docstring and explain all attributes
    def __init__(self, imp_readnetwork: readnetwork.ReadNetwork, imp_writenetwork: writenetwork.WriteNetwork,
                 imp_tube_ht: tubehaematocrit.TubeHaematocrit,
                 imp_tube_hd: dischargehaematocrit.DischargeHaematocrit,
                 imp_transmiss: transmissibility.Transmissibility, imp_buildsystem: buildsystem.BuildSystem,
                 imp_solver: pressureflowsolver.PressureFlowSolver, imp_rbcvelocity: rbc_velocity.RbcVelocity,
                 imp_iterative: iterative_routine.IterativeRoutine, imp_balance: flow_balance.FlowBalance,
                 PARAMETERS: MappingProxyType):
        # Network attributes
        self.hd_norm_plot = []
        self.bergIteration, self.Berg1, self.Berg2, self.BergPressure, self.BergFlow, self.BergHD, self.BergFirstPartEq, self.BergSecondPartEq = [], [], [], [], [], [], [], []
        self.vessel_flow_change_total = None
        self.maxBalance = None
        self.node_flow_change_total = None
        self.node_residual_plot = None
        self.node_relative_residual_plot = None
        self.vessel_flow_change = None
        self.positions_of_elements_not_in_boundary = None
        self.node_flow_change = None
        self.save_change_flow_over_th = None
        self.node_residual = None
        self.positions_of_elements_not_in_boundary = None
        self.two_MagnitudeThreshold = None
        self.local_balance_rbc = None
        self.min_flow = None
        self.eps_eff = None
        self.nr_of_vs = None
        self.nr_of_es = None

        # Vertex attributes
        self.xyz = None
        self.pressure = None

        # Edge attributes
        self.edge_list = None
        self.diameter = None
        self.length = None
        self.transmiss = None
        self.mu_rel = None
        self.ht = None
        self.hd = None
        self.flow_rate = None
        self.rbc_velocity = None

        # Connected Nodes
        self.edge_connected = None
        self.edge_connected_position = None
        self.node_connected = None

        # Network boundaries
        self.boundary_vs = None  # vertex ids of boundaries (1d np.array)
        self.boundary_val = None  # boundary values (1d np.array)
        self.boundary_type = None  # boundary type (1: pressure, 2: flow rate)

        # Solver
        self.system_matrix = None  # system matrix of linear system of equations
        self.rhs = None  # right hand side of linear system of equations

        # "References" to implementations
        self._imp_readnetwork = imp_readnetwork
        self._imp_writenetwork = imp_writenetwork
        self._imp_ht = imp_tube_ht
        self._imp_hd = imp_tube_hd
        self._imp_transmiss = imp_transmiss
        self._imp_buildsystem = imp_buildsystem
        self._imp_solver = imp_solver
        self._imp_rbcvelocity = imp_rbcvelocity
        self._imp_balance = imp_balance
        self.imp_iterative = imp_iterative

        # Threshold for zero-loops
        self.zeroFlowThreshold = None

        # "Reference" to parameter dict
        self._PARAMETERS = PARAMETERS

        self.alpha = PARAMETERS['alpha']
        self.sor = True

        self.avg_check = 0
        self.max_check = 0
        self.avg_old = 0
        self.iterationExit = 0
        self.hd_i = 0
        self.iteration = 0
        self.cnvg_rbc = 0
        self.cnvg_flow = 0
        self.alphaOn = True
        self.local_balance_rbc = True
        self.residualOverIteration, self.residualFlowOverIteration, self.i = [], [], 0
        self.residualOverIterationMax = []
        self.residualOverIterationNorm = []
        self.alphaSave = [1]
        self.stop, self.n_stop = False, 0
        self.all_positions = None
        self.node_values = None
        self.flagFlow, self.flagFlowM1 = None, None
        self.pressure_node, self.families_dict, self.vessel_general = None, None, None
        self.node_identifiers = [75, 193, 238, 377, 456, 522, 771, 778]  # MVN1_01
        # self.node_identifiers = [361, 405, 407, 576, 713, 950, 968, 1005, 2617]  # MVN2_06
        # self.node_identifiers = [300, 500]
        self.vessel_value_hd, self.vessel_value_flow = None, None
        self.node_values_hd, self.node_values_flow, self.upAlpha, self.max_magnitude, self.node_relative_residual = None, None, 0, 0, None
        self.zeroFlowThresholdMagnitude, self.indices_over, self.indices_over_blue, self.local_balance_rbc_corr = None, None, None, None
        self.boundary_inflow, self.families_dict_total = [], None
        self.increment = 0
        self.hd_convergence_criteria, self.flow_convergence_criteria = None, None
        self.hd_convergence_criteria_plot, self.flow_convergence_criteria_plot = [], []
        self.rasmussen_hd_threshold, self.rasmussen_flow_threshold = None, None
        self.hd_convergence_criteria_berg, self.flow_convergence_criteria_berg, self.pressure_convergence_criteria_berg = None, None, None
        self.inflow, self.inflow_pressure_node = None, None
        self.berg_criteria = 1e-6
        self.r_value = 10
        self.average_inlet_pressure, self.pressure_norm_plot = [], []
        return

    def read_network(self):
        """
        Read or import a vascular network.
        """
        self._imp_readnetwork.read(self)

    def write_network(self):
        """
        Write a vascular network to file
        """
        self._imp_writenetwork.write(self)

    def update_transmissibility(self):
        """
        Update transmissibility of all edges in the vascular network.
        """
        self._imp_ht.update_ht(self)
        self._imp_hd.update_hd(self)
        self._imp_transmiss.update_transmiss(self)

    def update_blood_flow(self):
        """
        Solve a linear system for updating pressures, flow rates and red blood cell velocities.
        """
        self._imp_buildsystem.build_linear_system(self)
        self._imp_solver.update_pressure_flow(self)
        self._imp_rbcvelocity.update_velocity(self)
        self.imp_iterative.iterative_function(self)

    def check_flow_balance(self):
        """
        Check flow balance
        """
        self._imp_balance.check_flow_balance(self)
