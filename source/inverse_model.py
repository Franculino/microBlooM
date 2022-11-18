from types import MappingProxyType
import source.flow_network as flow_network
import source.inverseproblemmethods.parameter_space as parameter_space


class InverseModel(object):
    # todo docstring and explain all attributes
    def __init__(self, flownetwork: flow_network.FlowNetwork, imp_parameterspace: parameter_space.ParameterSpace,
                 PARAMETERS: MappingProxyType):
        # "Reference" to flow network
        self._flow_network = flownetwork

        # "Reference" to parameter dict
        self._PARAMETERS = PARAMETERS

        # Parameter edge attributes
        self.alpha = None
        self.alpha_prime = None
        self.alpha_pm_range = None

        self.transmiss_base = None
        self.diameter_base = None

        self.mu_rel_tilde = None
        self.transmiss_tilde = None

        # Todo: account for parameters that can adjust

        # Inverse model parameters
        self.gamma = None
        self.phi = None

        # "References" to implementations
        self._imp_parameterspace = imp_parameterspace

    def initialise_inverse_model(self):
        # todo implementation
        self._imp_parameterspace.initialise_parameters(self, self._flow_network)


    def update_parameter(self):
        # todo implementation
        #  this is a construction site
        # update current cost function value (Class CostFunction)
        # get df/dT, df/dp (Class CostFunction)
        # get dg/dp and dg/dT (new class?)
        # get dT/d_alpha (Class ParameterSpace)
        # get d_alpha/d_alpha_prime (class AlphaMapping)
        # get gradient f (later maybe separate class?)
        # apply optimiser for alpha_prime and alpha (later: separate class, allow for Gauss-Newton also)
        # update d, T (Class ParameterSpace)
        pass





