import source.fileio.read_network as readnetwork
import source.fileio.write_network as writenetwork
import source.bloodflowmodel.tube_haematocrit as tubehaematocrit
import source.bloodflowmodel.discharge_haematocrit as dischargehaematocrit
import source.bloodflowmodel.transmissibility as transmissibility
import source.bloodflowmodel.pressure_flow_solver as pressureflowsolver
import source.bloodflowmodel.build_system as buildsystem
import source.bloodflowmodel.rbc_velocity as rbc_velocity
import source.distensibilitymodules.initialise_tube_law as initialise_tube_law
import source.fileio.read_vascular_properties as read_vascular_properties
from types import MappingProxyType


class FlowNetwork(object):
    # todo docstring and explain all attributes
    def __init__(self, imp_readnetwork: readnetwork.ReadNetwork, imp_writenetwork: writenetwork.WriteNetwork,
                 imp_tube_ht: tubehaematocrit.TubeHaematocrit,
                 imp_tube_hd: dischargehaematocrit.DischargeHaematocrit,
                 imp_transmiss: transmissibility.Transmissibility, imp_buildsystem: buildsystem.BuildSystem,
                 imp_solver: pressureflowsolver.PressureFlowSolver, imp_rbcvelocity: rbc_velocity.RbcVelocity,
                 imp_read_vascular_properties: read_vascular_properties.ReadVascularProperties,
                 imp_tube_law_ref_state: initialise_tube_law.TubeLawInitialision,
                 PARAMETERS: MappingProxyType):

        self.percent = None  #### REMOVE
        self.rel_stiffness = None  #### REMOVE
        self.rel_compliance = None  #### REMOVE
        self.shear_stress_baseline = None #### REMOVE
        self.sens_direct = None  #### REMOVE
        self.sens_shear = None  #### REMOVE

        # Network attributes
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

        # Network boundaries
        self.boundary_vs = None  # vertex ids of boundaries (1d np.array)
        self.boundary_val = None  # boundary values (1d np.array)
        self.boundary_type = None  # boundary type (1: pressure, 2: flow rate)

        # Solver
        self.system_matrix = None  # system matrix of linear system of equations
        self.rhs = None  # right hand side of linear system of equations

        # Tube law: Reference values
        self.pressure_ref = None
        self.diameter_ref = None
        self.pressure_external = None

        # Tube law: Material parameters (constants)
        self.e_modulus = None  # E modulus for each vessel with distensibility
        self.wall_thickness = None  # Vessel wall thickness
        self.nu = 0.5  # Poisson ratio of the vessel wall. nu = 0.5, if vessel walls are incompressible

        # "References" to implementations
        self._imp_readnetwork = imp_readnetwork
        self._imp_writenetwork = imp_writenetwork
        self._imp_ht = imp_tube_ht
        self._imp_hd = imp_tube_hd
        self._imp_transmiss = imp_transmiss
        self._imp_buildsystem = imp_buildsystem
        self._imp_solver = imp_solver
        self._imp_rbcvelocity = imp_rbcvelocity
        self._imp_tube_law_ref_state = imp_tube_law_ref_state
        self._imp_read_vascular_properties = imp_read_vascular_properties

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

    def initialise_tube_law(self):
        """
        Update transmissibility of all edges in the vascular network.
        """
        self._imp_read_vascular_properties.read(self)
        self._imp_tube_law_ref_state.initialise_ref_state(self)

