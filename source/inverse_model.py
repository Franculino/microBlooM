from types import MappingProxyType
import source.flow_network as flow_network
import source.inverseproblemmodules.adjoint_method_implementations as adj_method_parameters
import source.inverseproblemmodules.adjoint_method_solver as adj_method_solver
import source.inverseproblemmodules.alpha_restriction as alpha_mapping
import source.fileio.read_target_values as read_target_values
import source.fileio.read_parameters as read_parameters

import numpy as np


class InverseModel(object):

    def __init__(self, flownetwork: flow_network.FlowNetwork, imp_readtargetvalues: read_target_values.ReadTargetValues,
                 imp_readparameters: read_parameters.ReadParameters,
                 imp_adjointmethodparameters: adj_method_parameters.AdjointMethodImplementations,
                 imp_adjointmethodsolver: adj_method_solver.AdjointMethodSolver,
                 imp_alphamapping: alpha_mapping.AlphaRestriction, PARAMETERS: MappingProxyType):
        # "Reference" to flow network
        self._flow_network = flownetwork

        # "Reference" to parameter dict
        self._PARAMETERS = PARAMETERS

        # Target values
        self.edge_constraint_eid = None  # edge ids of constraint edges (1d np.array)
        self.edge_constraint_type = None  # constraint type - 1: Flow rate, 2: Velocity, ... (1d np.array)
        self.edge_constraint_value = None  # constraint value (1d np.array)
        self.edge_constraint_range_pm = None  # constraint range (1d np.array)
        self.edge_constraint_sigma = None  # constraint sigma of the cost function (1d np.array)
        self.nr_of_edge_constraints = None  # number of constraint edges

        # Parameter space
        # Edge parameters
        self.edge_param_eid = None  # edge ids of parameters (1d np.array)
        self.parameter_pm_range = None  # parameter range - tolerance to baseline (1d np.array)
        self.nr_of_edge_parameters = None  # number of edge parameters

        # Vertex parameters
        self.vertex_param_vid = None  # vetrex ids of parameters (1d np.array)
        # self.vertex_param_pm_range = None
        self.nr_of_vertex_parameters = None  # number of vertex parameters

        # Total parameters
        self.nr_of_parameters = None

        # Parameter edge and vertex attributes
        self.alpha = None
        self.alpha_prime = None
        self.alpha_pm_range = None

        self.transmiss_baselinevalue = None  # baseline transmissibility (1d np.array)
        self.diameter_baselinevalue = None  # baseline diameter (1d np.array)

        self.boundary_pressure_baselinevalue = None

        self.mu_rel_tilde = None
        self.transmiss_tilde = None

        # Inverse model parameters
        self.gamma = None  # constant learning rate
        self.phi = None  # constant parameter to tune the shape of the function between alpha and alpha_prime

        # Inverse model cost terms
        self.f_h = None  # Cost of hard constraint

        # Adjoint method vectors and matrices - derivatives
        self.d_f_d_alpha = None  # Vector
        self.d_f_d_pressure = None  # Vector
        self.d_g_d_alpha = None  # coo_matrix
        self._lambda = None  # Vector

        # Gradient descent
        self.gradient_alpha = None
        self.gradient_alpha_prime = None

        # "References" to implementations
        self._imp_adjointmethodparameters = imp_adjointmethodparameters
        self._imp_readtargetvalues = imp_readtargetvalues
        self._imp_readparameters = imp_readparameters
        self._imp_adjointmethodsolver = imp_adjointmethodsolver
        self._imp_alphamapping = imp_alphamapping

        # Simulation monitoring - Visualisation
        self.current_iteration = 0
        self.iteration_array = np.array([])
        self.f_h_array = np.array([])

    def initialise_inverse_model(self):
        """
        Method to initialise the inverse model based on target values and defined parameters.
        """
        self._imp_readtargetvalues.read(self, self._flow_network)
        self._imp_readparameters.read(self, self._flow_network)
        self._imp_adjointmethodparameters.initialise_parameters(self, self._flow_network)
        self.gamma = self._PARAMETERS["gamma"]
        self.phi = self._PARAMETERS["phi"]

    def update_state(self):
        """
        Method to update the parameter vector alpha with constant step-width gradient descent.
        """
        # Update all partial derivatives needed to solve the adjoint method
        self._imp_adjointmethodparameters.update_partial_derivatives(self, self._flow_network)
        # Update gradient d f / d alpha by solving the adjoint method
        self._imp_adjointmethodsolver.update_gradient_alpha(self, self._flow_network)
        # Update gradient d f / f alpha_prime (mapping between parameter and pseudo parameter)
        self._imp_alphamapping.update_gradient_alpha_prime(self)
        # Update alpha_prime by using gradient descent with constant learning rate.
        self.alpha_prime -= self.gamma * self.gradient_alpha_prime
        # Transform pseudo parameter alpha_prime back to alpha space
        self._imp_alphamapping.update_alpha_from_alpha_prime(self)
        # Update the system state depending on the parameter (e.g. diameter, transmissibility, boundary pressures)
        self._imp_adjointmethodparameters.update_state(self, self._flow_network)

    def update_cost(self):
        """
        Method to update the cost function value
        """
        # Update the cost function value
        self._imp_adjointmethodparameters.update_cost_hardconstraint(self, self._flow_network)