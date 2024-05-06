from types import MappingProxyType
import source.flow_network as flow_network
import source.distensibility as distensibility
import source.fileio.read_autoregulation_parameters as read_autoregulation_parameters
import source.autoregulationmodules.initialise_autoregulation_model as initialise_autoregulation_model
import source.autoregulationmodules.update_diam_autoregulation_model as update_diam_autoregulation_model

class Autoregulation(object):
    def __init__(self, flownetwork: flow_network.FlowNetwork,
                 imp_read_auto_parameters: read_autoregulation_parameters.ReadAutoregulationParameters,
                 imp_auto_baseline: initialise_autoregulation_model.AutoregulationModelInitialise,
                 imp_auto_feedback_model: update_diam_autoregulation_model.AutoregulationModelUpdate):

        # "Reference" to flow network
        self._flow_network = flownetwork

        # Modelling parameters
        self.nr_of_edge_autoregulation = None  # Number of vessels that actively change their diameters (autoregulatory vessels)
        self.eid_vessel_autoregulation = None  # Edge ids of vessels that actively change their diameters (autoregulatory vessels)
        self.sens_direct = None  # sensitivity for the direct stress, Sσ
        self.sens_shear = None  # sensitivity for the shear stress, Sτ
        self.sens_shear_previous = None

        self.rel_stiffness = None  # relative stiffness
        self.rel_compliance = None  # relative compliance

        # Baseline values form the flow network model
        self.diameter_baseline = None  # Baseline diameter for the entire network
        self.pressure_baseline = None  # Baseline pressure for the entire network
        self.flow_rate_baseline = None  # Baseline flow rate for the entire network
        self.pressure_dif_base = None  # Pressure different between baseline pressure and external pressure

        # Reference values for the p-A relationship (D_ref or A_ref)
        self.diameter_ref = None  # Reference diameter of the autoregulatory vessels

        # Parameters for the relaxation function
        self.diameter_previous = None  # previous diameter for the entire network
        self.alpha = None  # relaxation factor

        # Baseline values form the autoregulatory model
        self.compliance_baseline = None  # Baseline compliance
        self.direct_stress_baseline = None  # Baseline direct stress
        self.shear_stress_baseline = None  # Baseline shear stress

        self.direct_stress = None  # Direct stress
        self.shear_stress = None  # Shear stress
        self.compliance = None  # compliance

        # Loop - Compliance feedback model
        self.iteration = None  # iterations for the compliance feedback model
        self.direct_stress_previous = None
        self.volume_previous = None

        # "References" to implementations
        self._imp_read_auto_parameters = imp_read_auto_parameters
        self._imp_auto_baseline = imp_auto_baseline
        self._imp_auto_feedback_model = imp_auto_feedback_model

    def initialise_autoregulation(self):
        """
        Method to initialise the autoregulation model.
        """
        self._imp_read_auto_parameters.read(self, self._flow_network)
        self._imp_auto_baseline.initialise_baseline_stresses(self, self._flow_network)
        self._imp_auto_baseline.initialise_baseline_compliance(self, self._flow_network)

    def update_vessel_diameters_auto(self):
        """
        Method to update the  diameters based on different p-A relations (distensibility law)
        """
        self._imp_auto_feedback_model.update_diameter(self, self._flow_network)
