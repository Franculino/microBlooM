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

        # Iterative procedure
        self.alpha = None
        self.sor = True
        # the number of iterations performed
        # 1st (0) is to stabilize the iteration
        self.iteration = 0
        # Zero-flow threshold
        self.zeroFlowThresholdMagnitude = None
        # Threshold terative approach with our sor
        self.two_MagnitudeThreshold = None
        # save residual max and norm over the iteration
        self.residualOverIterationMax = []
        self.residualOverIterationNorm = []
        # to save the value of Alpha's during the iteration and display in the final plot
        self.alphaSave = [1]
        # to stop our computation after a certain iteration
        self.stop = False
        # array for discharge hematicrit
        self.pressure_node = None
        # assistant variables for an iterative process over 4000 iteration
        self.vessel_general = None
        self.vessel_value_hd = None
        self.vesel_flow_change = None
        self.vessel_value_flow = None
        self.node_values_hd = None
        self.node_values_flow = None
        self.node_relative_residual = None
        self.indices_over = None
        self.indices_over_blue = None
        self.families_dict_total = None
        # Convergence criteria for iterative approaches Berg/Rasmussen
        self.hd_convergence_criteria = None
        self.flow_convergence_criteria = None
        self.hd_convergence_criteria_berg = None
        self.flow_convergence_criteria_berg = None
        self.pressure_convergence_criteria_berg = None
        self.berg_criteria = 1e-6
        self.r_value = 10
        # Threshold for Rasmussen approach
        self.rasmussen_hd_threshold = None
        self.rasmussen_flow_threshold = None
        # Assistence for iterative approaches
        self.inflow = None
        self.inflow_pressure_node = None
        self.average_inlet_pressure = []
        self.bergIteration = []
        self.Berg1 = []
        self.Berg2 = []
        self.BergPressure = []
        self.BergFlow = []
        self.BergHD = []
        self.BergFirstPartEq = []
        self.BergSecondPartEq = []
        self.vessel_flow_change_total = None
        self.maxBalance = None
        self.node_flow_change_total = None
        self.vessel_flow_change = None
        self.node_flow_change = None
        self.node_residual = None
        self.positions_of_elements_not_in_boundary = None
        self.local_balance_rbc = None

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

    # Needed for Iterative procedure
    @property
    def imp_buildsystem(self):
        return self._imp_buildsystem

    # Needed for Iterative procedure
    @property
    def imp_solver(self):
        return self._imp_solver

    # Needed for Iterative procedure
    @property
    def imp_rbcvelocity(self):
        return self._imp_rbcvelocity
