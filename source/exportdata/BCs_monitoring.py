import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from types import MappingProxyType
import source.flow_network as flow_network
import source.inverse_model as inverse_model


class BCs_monitoring(object):
    """
    Class for monitoring the solution for the bc_tuning model.
    """

    def __init__(self, flownetwork: flow_network.FlowNetwork, inversemodel: inverse_model.InverseModel,
                 PARAMETERS: MappingProxyType):

        self.flownetwork = flownetwork
        self.inversemodel = inversemodel
        self._PARAMETERS = PARAMETERS

    def get_arrays(self):
        """
        Arrays preparation for plotting the pressure BCs vs iterations.
        """
        current_iteration = self.inversemodel.current_iteration

        iteration_array = np.arange(1, current_iteration + 1)
        BCs_pressure = self.inversemodel.alpha
        self.inversemodel.BCs_matrix = (np.append(self.inversemodel.BCs_matrix, np.vstack(BCs_pressure))
                                        .reshape(-1, np.size(self.inversemodel.vertex_param_vid)))

        return iteration_array

    def BCs_csv(self):
        """
        Save the evolution of the boundary condition values throughout the iterations
        """
        csv_path = self._PARAMETERS["csv_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration

        filepath_bcs_pressure = (csv_path + "gamma_" + str(self._PARAMETERS["gamma"]) +
                                 "_targets_" + str(self._PARAMETERS["n_targets"]) + "_trial_bcs_BCs_pressure_" +
                                 str(current_iteration) + ".csv")

        df_BCs = pd.DataFrame(self.inversemodel.BCs_matrix)
        df_BCs.to_csv(filepath_bcs_pressure)

        return

    def plot_BCs_vs_iterations(self, iteration_array):
        """
        Plot the evolution of the boundary condition values throughout the iterations
        """
        png_path = self._PARAMETERS["png_path_solution_monitoring"]
        current_iteration = self.inversemodel.current_iteration
        filepath_png = (png_path + "gamma_" + str(self._PARAMETERS["gamma"]) +
                        "_targets_" + str(self._PARAMETERS["n_targets"]) + "_trial_BCs_vs_iterations_" +
                        str(current_iteration) + ".png")

        plt.figure(figsize=(10, 8))
        plt.plot(iteration_array, self.inversemodel.BCs_matrix)
        plt.title('BCs vs. Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('BCs pressures')
        plt.grid(True)

        # Save the figure in a png file
        plt.savefig(filepath_png, dpi=600)

        return
