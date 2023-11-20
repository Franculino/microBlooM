from types import MappingProxyType
import source.flow_network as flow_network
import source.distensibility as distensibility
import source.fileio.read_autoregulation_parameters as read_autoregulation_parameters
import source.autoregulationmodules.initialise_autoregulation_model as initialise_autoregulation_model
import source.autoregulationmodules.update_diam_autoregulation_model as update_diam_autoregulation_model

class Autoregulation(object):
    def __init__(self, flownetwork: flow_network.FlowNetwork, distensibility: distensibility.Distensibility,
                 imp_read_auto_parameters: read_autoregulation_parameters.ReadAutoregulationParameters,
                 imp_auto_baseline: initialise_autoregulation_model.AutoregulationModelInitialise,
                 imp_auto_feedback_model: update_diam_autoregulation_model.AutoregulationModelUpdate):

        # "Reference" to flow network
        self._flow_network = flownetwork
        self._distensibility = distensibility

        # Modelling parameters
        self.nr_of_edge_autoregulation = None  # Number of vessels that actively change their diameters (autoregulatory vessels)
        self.eid_vessel_autoregulation = None  # Edge ids of vessels that actively change their diameters (autoregulatory vessels)
        self.e_modulus = None  # Young's Modulus of the autoregulatory vessels
        self.wall_thickness = None  # Wall thickness of the autoregulatory vessels
        self.sens_direct = None  # sensitivity for the direct stress, Sσ
        self.sens_shear = None  # sensitivity for the shear stress, Sτ


        # Baseline values form the flow network model
        self.diameter_baseline = None  # Baseline diameter
        self.pressure_baseline = None  # Baseline pressure
        self.flow_rate_baseline = None

        # Baseline values form the autoregulatory model
        self.compliance_baseline = None  # Baseline compliance
        self.direct_stress_baseline = None  # Baseline direct stress
        self.shear_stress_baseline = None  # Baseline shear stress

        self.compliance = None  # compliance

        # "References" to implementations
        self._imp_read_auto_parameters = imp_read_auto_parameters
        self._imp_auto_baseline = imp_auto_baseline
        self._imp_auto_feedback_model = imp_auto_feedback_model

    def initialise_autoregulation(self):
        """
        Method to initialise the autoregulation model.
        """
        self._imp_read_auto_parameters.read(self, self._distensibility, self._flow_network)
        self._imp_auto_baseline.initialise_baseline_stresses(self, self._flow_network)
        self._imp_auto_baseline.initialise_baseline_compliance(self, self._flow_network)

    def update_vessel_diameters_auto(self):
        """
        Method to update the  diameters based on different p-A relations (distensibility law)
        """
        self._imp_auto_feedback_model.update_diameter(self, self._flow_network)
