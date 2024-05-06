from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class ReadAutoregulationParameters(ABC):
    """
    Abstract base class for reading the parameters related to the autoregulatory blood vessels
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of ReadAutoregulationParameters.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def read(self, autoregulation, flownetwork):
        """
        Import eids with distensibility parameters, sensitivity factors of the direct (σ) and shear (τ) stress, ...
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class ReadAutoregulationParametersNothing(ReadAutoregulationParameters):
    """
    Class for not reading any parameters related to the autoregulatory blood vessels
    """

    def read(self, autoregulation, flownetwork):
        """
        Do not import any parameters
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

class ReadAutoregulationParametersFromFile(ReadAutoregulationParameters):
    """
    Class for reading the parameters related to the autoregulatory blood vessels from a csv file
    """

    def read(self, autoregulation, flownetwork):
        """
        Import eids with distensibility parameters, sensitivity factors of the direct (σ) and shear (τ) stress, ...
        :param autoregulation: autoregulation object
        :type autoregulation: source.autoregulation.Autoregulation
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        import pandas as pd
        # Extract file path of autoregulation parameters.
        path_autoregulation_parameters = self._PARAMETERS["csv_path_autoregulation"]

        # Read file with pandas
        df_autoregulation = pd.read_csv(path_autoregulation_parameters)

        # Check for keyword all. If all is found in first line, the identical autoregulation parameters are used for
        # all edges
        if df_autoregulation['eid_autoregulation'][0] == 'all':
            autoregulation.eid_vessel_autoregulation = np.arange(flownetwork.nr_of_es, dtype=int)
            autoregulation.sens_direct = np.ones(flownetwork.nr_of_es, dtype=float) * \
                                         df_autoregulation['sensitivity_direct_stress'][0]  # Sσ
            autoregulation.sens_shear = np.ones(flownetwork.nr_of_es, dtype=float) * \
                                        df_autoregulation['sensitivity_shear_stress'][0]  # Sτ

        else:
            # Sort prescribed edge ids with autoregulation according to ascending edge ids.
            df_autoregulation = df_autoregulation.sort_values('eid_autoregulation')
            # Check for duplicate eids
            if True in df_autoregulation.duplicated(subset=['eid_autoregulation']).to_numpy():
                sys.exit("Error: Duplicate edge id in autoregulation definition.")

            autoregulation.eid_vessel_autoregulation = df_autoregulation["eid_autoregulation"].to_numpy().astype(int)
            # autoregulation.sens_direct = df_autoregulation['sensitivity_direct_stress'].to_numpy().astype(float) * \
            #                              self._PARAMETERS["sensitivity_direct_stress"]
            # autoregulation.sens_shear = df_autoregulation['sensitivity_shear_stress'].to_numpy().astype(float) * \
            #                              self._PARAMETERS["sensitivity_shear_stress"]

        autoregulation.nr_of_edge_autoregulation = np.size(autoregulation.eid_vessel_autoregulation)
