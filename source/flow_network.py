import source.fileio.read_network as readnetwork
import source.fileio.write_network as writenetwork
import source.bloodflowmodel.tube_haematocrit as tubehaematocrit
import source.bloodflowmodel.discharge_haematocrit as dischargehaematocrit
import source.bloodflowmodel.transmissibility as transmissibility
import source.bloodflowmodel.pressure_flow_solver as pressureflowsolver
import source.bloodflowmodel.build_system as buildsystem
import source.bloodflowmodel.rbc_velocity as rbc_velocity
from types import MappingProxyType


class FlowNetwork(object):

    def __init__(self, imp_readnetwork: readnetwork.ReadNetwork, imp_writenetwork: writenetwork.WriteNetwork,
                 imp_tube_ht: tubehaematocrit.TubeHaematocrit,
                 imp_tube_hd: dischargehaematocrit.DischargeHaematocrit,
                 imp_transmiss: transmissibility.Transmissibility, imp_buildsystem: buildsystem.BuildSystem,
                 imp_solver: pressureflowsolver.PressureFlowSolver, imp_rbcvelocity: rbc_velocity.RbcVelocity,
                 PARAMETERS: MappingProxyType):

        # Network attributes
        self.nr_of_vs = None  # number of vertices
        self.nr_of_es = None  # number of edges

        # Vertex attributes
        self.xyz = None  # coordination (2d np.array)
        self.pressure = None  # pressure (1d np.array)

        # Edge attributes
        self.edge_list = None  # edge list (2d np.array)
        self.diameter = None  # diameter (1d np.array)
        self.length = None  # length (1d np.array)
        self.transmiss = None  # transmissibility (1d np.array)
        self.mu_rel = None  # relative apparent viscosity (1d np.array)
        self.ht = None  # tube haematocrit (1d np.array)
        self.hd = None  # discharge haematocrit (1d np.array)
        self.flow_rate = None  # flow rate (1d np.array)
        self.rbc_velocity = None  # rbc velocity (1d np.array)

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

        # "Reference" to parameter dict
        self._PARAMETERS = PARAMETERS
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


