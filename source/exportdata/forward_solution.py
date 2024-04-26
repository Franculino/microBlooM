import pandas as pd
from types import MappingProxyType
import source.flow_network as flow_network


class forward_problem_solution(object):
    """
    Class for monitoring the solution for the forward model
    """

    def __init__(self, flownetwork: flow_network.FlowNetwork, PARAMETERS: MappingProxyType):

        self.flownetwork = flownetwork
        self._PARAMETERS = PARAMETERS

    def flow_rate_csv(self):
        """
        Save the flow rates to a CSV file.
        """
        csv_path = self._PARAMETERS["flow_rate_path"]
        flow_rate = self.flownetwork.flow_rate
        df_flow_rate = pd.DataFrame({'flow_rate': pd.Series(flow_rate)})
        df_flow_rate.to_csv(csv_path, index=False)

    def pressures_csv(self):
        csv_path = self._PARAMETERS["pressure_path"]
        pressure = self.flownetwork.pressure
        df_pressures = pd.DataFrame({'pressures': pd.Series(pressure)})
        df_pressures.to_csv(csv_path, index=False)

        return

    def rbc_velocity_csv(self):
        csv_path = self._PARAMETERS["rbc_velocity_path"]
        rbc_velocities = self.flownetwork.rbc_velocity
        df_rbc_velocity = pd.DataFrame({'rbc_velocities': pd.Series(rbc_velocities)})
        df_rbc_velocity.to_csv(csv_path, index=False)

        return

