from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class ReadDistensibilityParameters(ABC):
    """
    Abstract base class for reading the parameters related to the distensibility of blood vessels
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of ReadParameters.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def read(self, distensibility, flownetwork):
        """
        Import...
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class ReadDistensibilityParametersNothing(ReadDistensibilityParameters):
    """
    Class for not reading any parameters related to the distensibility of blood vessels
    """

    def read(self, distensibility, flownetwork):
        """
        Import...
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

class ReadDistensibilityParametersFromFile(ReadDistensibilityParameters):
    """
    Class for reading the parameters related to the distensibility of blood vessels from a file
    """

    def read(self, distensibility, flownetwork):
        """
        Import...
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        import pandas as pd
        # Extract file path of distensibility parameters.
        path_distensibility_parameters = self._PARAMETERS["csv_path_distensibility"]

        # Read file with pandas
        df_distensibility = pd.read_csv(path_distensibility_parameters)

        # Check for keyword all. If all is found in first line, the identical distensibility parameters are used for
        # all edges
        if df_distensibility['eid_distensibility'][0] == 'all':
            distensibility.eid_vessel_distensibility = np.arange(flownetwork.nr_of_es, dtype=int)
            distensibility.e_modulus = np.ones(flownetwork.nr_of_es, dtype=float) * df_distensibility['e_modulus'][0]
            distensibility.wall_thickness = np.ones(flownetwork.nr_of_es, dtype=float) * \
                                            df_distensibility['wall_thickness'][0]
        else:
            # Sort prescribed edge ids with distensibility according to ascending edge ids.
            df_distensibility = df_distensibility.sort_values('eid_distensibility')
            # Check for duplicate eids
            if True in df_distensibility.duplicated(subset=['eid_distensibility']).to_numpy():
                sys.exit("Error: Duplicate edge id in distensibility definition.")

            distensibility.eid_vessel_distensibility = df_distensibility["eid_distensibility"].to_numpy().astype(int)
            distensibility.e_modulus = df_distensibility['e_modulus'].to_numpy().astype(float)
            distensibility.wall_thickness = df_distensibility['wall_thickness'].to_numpy().astype(float)

            if np.max(distensibility.eid_vessel_distensibility) > flownetwork.nr_of_es - 1:
                sys.exit("Error: Distensibility refers to invalid edge id.")

        distensibility.nr_of_edge_distensibilities = np.size(distensibility.eid_vessel_distensibility)
