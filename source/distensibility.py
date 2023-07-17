from types import MappingProxyType
import source.flow_network as flow_network
import source.distensibilitymodules.distensibility_law as distensibility_law
import source.fileio.read_distensibility_parameters as read_distensibility_parameters


class Distensibility(object):
    def __init__(self, flownetwork: flow_network.FlowNetwork,
                 imp_distensibility_ref_state: distensibility_law.DistensibilityLawInitialise,
                 imp_read_distensibility_parameters: read_distensibility_parameters.ReadDistensibilityParameters,
                 imp_distensibility_relation: distensibility_law.DistensibilityLawUpdate):
        # "Reference" to flow network
        self._flow_network = flownetwork

        # Modelling constants
        self.nr_of_edge_distensibilities = None
        self.e_modulus = None  # E modulus for each vessel with distensibility
        self.wall_thickness = None  # Vessel wall thickness
        self.eid_vessel_distensibility = None  # Eids that have a vessel distensibility
        self.nu = 0.5  # Poisson ratio of the vessel wall. nu = 0.5, if vessel walls are incompressible
        self.pressure_external = None

        # Reference values
        self.pressure_ref = None
        self.diameter_ref = None

        # "References" to implementations
        self._imp_distensibility_ref_state = imp_distensibility_ref_state
        self._imp_read_distensibility_parameters = imp_read_distensibility_parameters
        self._imp_distensibility_relation = imp_distensibility_relation

    def initialise_distensibility(self):
        """
        Method to initialise the distensibility model.
        """
        self._imp_read_distensibility_parameters.read(self, self._flow_network)
        self._flow_network.update_transmissibility()
        self._flow_network.update_blood_flow()
        self._imp_distensibility_ref_state.initialise_distensibility_ref_state(self, self._flow_network)

    def update_vessel_diameters(self):
        """
        Method to update the  diameters based on different distensibility p-A relations
        """
        self._imp_distensibility_relation.update_diameter(self, self._flow_network)
