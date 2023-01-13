from types import MappingProxyType
import source.flow_network as flow_network
import source.inverseproblemmethods.parameter_space as parameter_space
import source.fileio.read_target_values as read_target_values
import source.fileio.read_parameters as read_parameters


class InverseModel(object):
    # todo docstring and explain all attributes
    def __init__(self, flownetwork: flow_network.FlowNetwork, imp_readtargetvalues: read_target_values.ReadTargetValues,
                 imp_readparameters: read_parameters.ReadParameters,
                 imp_parameterspace: parameter_space.ParameterSpace, PARAMETERS: MappingProxyType):
        # "Reference" to flow network
        self._flow_network = flownetwork

        # "Reference" to parameter dict
        self._PARAMETERS = PARAMETERS

        # Target values
        self.edge_constraint_eid = None
        self.edge_constraint_type = None # 1: Flow rate, 2: Velocity, ...
        self.edge_constraint_value = None
        self.edge_constraint_range_pm = None
        self.edge_constraint_sigma = None
        self.nr_of_edge_constraints = None

        # Parameter space
        # Edge parameters
        self.edge_param_eid = None
        self.edge_param_pm_range = None
        self.nr_of_edge_parameters = None

        # Vertex parameters
        self.vertex_param_vid = None
        self.vertex_param_pm_range = None
        self.nr_of_vertex_parameters = None

        # Parameter edge attributes
        self.alpha = None
        self.alpha_prime = None
        self.alpha_pm_range = None

        self.transmiss_base = None
        self.diameter_baselinevalue = None

        self.mu_rel_tilde = None
        self.transmiss_tilde = None

        # Inverse model parameters
        self.gamma = None
        self.phi = None

        # "References" to implementations
        self._imp_parameterspace = imp_parameterspace
        self._imp_readtargetvalues = imp_readtargetvalues
        self._imp_readparameters = imp_readparameters

    def initialise_inverse_model(self):
        # todo implementation
        self._imp_readtargetvalues.read(self)
        self._imp_readparameters.read(self)
        self._imp_parameterspace.initialise_parameters(self, self._flow_network)


    def update_parameter(self):
        # todo implementation
        # this is a construction site
        # update current cost function value (Class CostFunction)
        # get df/dT, df/dp (Class CostFunction)
        # get dg/dp and dg/dT (new class?)
        # get dT/d_alpha (Class ParameterSpace)
        # get d_alpha/d_alpha_prime (class AlphaMapping)
        # get gradient f (later maybe separate class?)
        # apply optimiser for alpha_prime and alpha (later: separate class, allow for Gauss-Newton also)
        # update d, T (Class ParameterSpace)
        pass





