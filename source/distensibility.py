from types import MappingProxyType
import source.flow_network as flow_network
import source.distensibilitymodules.distensibility_law as distensibility_law
import source.fileio.read_distensibility_parameters as read_distensibility_parameters



class Distensibility(object):
    def __init__(self, flownetwork: flow_network.FlowNetwork,
                 imp_distensibilitylaw: distensibility_law.DistensibilityLaw,
                 imp_read_distensibility_parameters: read_distensibility_parameters.ReadDistensibilityParameters):

        # "Reference" to flow network
        self._flow_network = flownetwork

        # Modelling constants
        self.nr_of_edge_distensibilities = None
        self.e_modulus = None  # E modulus for each vessel with distensibility
        self.wall_thickness = None  # Vessel wall thickness
        self.eid_vessel_distensibility = None  # Eids that have a vessel distensibility
        self.nu = 0.5  # Poisson ratio of the vessel wall. nu = 0.5, if vessel walls are incompressible
        self.pressure_external = None  # Extravascular pressure

        # Reference values
        self.pressure_ref = None
        self.diameter_ref = None

        # "References" to implementations
        self._imp_distensibility_law = imp_distensibilitylaw
        self._imp_read_distensibility_parameters = imp_read_distensibility_parameters

    def initialise_distensibility(self):
        """
        Method to initialise the distensibility model.
        """
        self._imp_read_distensibility_parameters.read(self, self._flow_network)
        self._flow_network.update_transmissibility()
        self._flow_network.update_blood_flow()
        self._imp_distensibility_law.initialise_distensibility_law(self, self._flow_network)

    def update_vessel_diameters(self):
        """
        Method to blablabla
        """
        self._imp_distensibility_law.update_diameter(self, self._flow_network)
