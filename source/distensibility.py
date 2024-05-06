from types import MappingProxyType
import source.flow_network as flow_network
import source.distensibilitymodules.update_diam_distensibility_law as update_diam_distensibility_law
import source.fileio.read_distensibility_parameters as read_distensibility_parameters


class Distensibility(object):
    def __init__(self, flownetwork: flow_network.FlowNetwork,
                 imp_read_dist_parameters: read_distensibility_parameters.ReadDistensibilityParameters,
                 imp_dist_pres_area_relation: update_diam_distensibility_law.DistensibilityLawUpdate):

        # "Reference" to flow network
        self._flow_network = flownetwork

        # Modelling constants
        self.nr_of_edge_distensibilities = None
        self.eid_vessel_distensibility = None  # Eids that have a vessel distensibility

        # Reference values
        self.pressure_ref = None
        self.diameter_ref = None

        # "References" to implementations
        self._imp_read_dist_parameters = imp_read_dist_parameters
        self._imp_dist_pres_area_relation = imp_dist_pres_area_relation

    def initialise_distensibility(self):
        """
        Method to initialise the distensibility model.
        """
        self._imp_read_dist_parameters.read(self, self._flow_network)

    def update_vessel_diameters_dist(self):
        """
        Method to update the  diameters based on different p-A relations (distensibility law)
        """
        self._imp_dist_pres_area_relation.update_diameter(self, self._flow_network)
