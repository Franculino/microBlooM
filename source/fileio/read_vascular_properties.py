from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class ReadVascularProperties(ABC):
    """
    Abstract base class for reading the parameters related to the vascular properties of blood vessels
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of ReadVascularProperties.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def read(self, flownetwork):
        """
        Import the Young's modulus, vascular wall thickness, ...
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class ReadVascularPropertiesNothing(ReadVascularProperties):
    """
    Class for not reading any parameters related to the vascular properties of blood vessels
    """

    def read(self, flownetwork):
        """
        Do not import any parameters
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class ReadVascularPropertiesFromFile(ReadVascularProperties):
    """
    Class for reading the parameters related to the vascular properties of blood vessels from a file
    """

    def read(self, flownetwork):
        """
        Import the Young's modulus, vascular wall thickness, ...
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        import pandas as pd
        # Extract file path of vascular properties
        path_vascular_properties = self._PARAMETERS["csv_path_vascular_properties"]

        # Read file with pandas
        df_vascular_properties = pd.read_csv(path_vascular_properties)

        # Check for keyword all. If all is found in first line, the identical vascular properties are used for
        # all edges
        if df_vascular_properties['eid'][0] == 'all':
            flownetwork.e_modulus = np.ones(flownetwork.nr_of_es, dtype=float) * df_vascular_properties['e_modulus'][0]
            flownetwork.wall_thickness = np.ones(flownetwork.nr_of_es, dtype=float) * \
                                            df_vascular_properties['wall_thickness'][0]
        else:
            # Sort prescribed edge ids according to ascending edge ids.
            df_vascular_properties = df_vascular_properties.sort_values('eid')
            # Check for duplicate eids
            if True in df_vascular_properties.duplicated(subset=['eid']).to_numpy():
                sys.exit("Error: Duplicate edge id in tube law definition.")

            eid_vessel = df_vascular_properties["eid"].to_numpy().astype(int)

            flownetwork.e_modulus = df_vascular_properties['e_modulus'].to_numpy().astype(float)
            flownetwork.wall_thickness = df_vascular_properties['wall_thickness'].to_numpy().astype(float)

            if np.max(eid_vessel) > flownetwork.nr_of_es - 1:
                sys.exit("Error: Tube law refers to invalid edge id.")
