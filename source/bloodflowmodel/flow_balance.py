import numpy as np
import matplotlib.pyplot as plt
import source.flow_network as flow_network
from types import MappingProxyType
import seaborn as sns


class FlowBalance(object):

    def __init__(self, flownetwork: flow_network.FlowNetwork):

        self.flownetwork = flownetwork
        #self._PARAMETERS = PARAMETERS

    def _get_flow_balance(self):

        nr_of_vs = self.flownetwork.nr_of_vs
        nr_of_es = self.flownetwork.nr_of_es

        edge_list = self.flownetwork.edge_list

        flow_balance = np.zeros(nr_of_vs)
        flow_rate = self.flownetwork.flow_rate

        for eid in range(nr_of_es):
            flow_balance[edge_list[eid, 0]] += flow_rate[eid]
            flow_balance[edge_list[eid, 1]] -= flow_rate[eid]

        return flow_balance

    def percentage_lower_than(self, data, value):
        """
        Calculate the percentage of elements in the dataset that are lower than the given value.

        Parameters:
            data (numpy.ndarray): The dataset.
            value (float): The value to compare with.

        Returns:
            float: The percentage of elements lower than the given value.
        """
        lower_count = np.sum(data < value)
        total_count = len(data)
        percentage = (lower_count / total_count) * 100
        return percentage

    def frequency_plot(self, data, title, x_axis, color_plot):
        mean_val = np.mean(data)
        median_val = np.median(data)
        max_val = np.max(data)
        print("The value for " + str(title) + " of: median = " + str(median_val) + " mean = " + str(mean_val) + " min = " + str(np.min(data[data != 0])) + " max = " + str(max_val))

        percentage = self.percentage_lower_than(np.abs(data), self.flownetwork.tolerance)
        print(f"The percentage of elements lower than {self.flownetwork.tolerance} is: {percentage:.2f}%")
        # Create the histogram with seaborn
        bin_count = 20000
        # Step 1: Determine the interval where you want to increase the number of bins
        lower_bound = 0  # Replace with the lower value of the interval
        upper_bound = 1e-16  # Replace with the upper value of the interval
        extra_bins = 100  # Number of extra bins in the specified interval

        interval_data = data[(data >= lower_bound) & (data <= upper_bound)]
        custom_bin_edges = np.linspace(lower_bound, upper_bound, extra_bins + 1)

        sns.histplot(interval_data, bins=custom_bin_edges, kde=False, color=color_plot, edgecolor='white', stat="percent")

        sns.histplot(data[(data < lower_bound) | (data > upper_bound)], bins=bin_count, kde=False, color=color_plot, edgecolor='white', stat="percent")

        plt.xlabel(x_axis)
        plt.ylabel('Percentage (%)')  # Update y-axis label
        plt.title(title)
        sns.despine(left=True)  # Remove the left and top spines
        plt.grid(axis='y', alpha=0.5)  # Add gridlines to the y-axis
        plt.xticks(fontsize=8)  # Adjust x-axis tick font size

        plt.tick_params(axis='y', which='both', color='#f0f0f0')

        # Calculate bin heights (frequencies) and total data points
        bin_heights, _ = np.histogram(data, bins=bin_count)

        plt.axvline(mean_val, color='red', linestyle='dashed', label='Mean', linewidth=1)
        plt.axvline(median_val, color='blue', linestyle='dashed', label='Median', linewidth=1)
        plt.axvline(max_val, color='green', linestyle='dashed', label='Max', linewidth=1)
        plt.axvline(self.flownetwork.tolerance, color='orange', linestyle='dashed', label='Tolerance', linewidth=1)

        # Log y-axis
        plt.xscale('log')

        # Add a legend
        plt.legend(loc='upper right')
        # Set y-axis tick locations and labels

        # Add a background color to the plot area
        plt.gca().set_facecolor('#f0f0f0')
        plt.tight_layout()
        plt.show()

    def calculate_percentile(self, data, percentile):
        # Step 1: Sort the data in ascending order
        sorted_data = sorted(data)
        # Step 2: Determine the position (rank) of the value within the sorted data
        n = len(sorted_data)
        position = (percentile / 100) * (n - 1)
        # Step 3: Calculate the percentile using interpolation
        if position.is_integer():  # If the position is an integer, no interpolation needed
            return sorted_data[int(position)]
        else:
            lower_index = int(position)
            upper_index = lower_index + 1
            lower_value = sorted_data[lower_index]
            upper_value = sorted_data[upper_index]
            return lower_value + (position - lower_index) * (upper_value - lower_value)

    def find_approx_percentile_of_value(self,data, value):
        # Step 1: Sort the data in ascending order
        sorted_data = sorted(data)

        # Step 2: Find the index where the value would be inserted in the sorted data
        index = 0
        while index < len(sorted_data) and sorted_data[index] < value:
            index += 1

        # Step 3: Calculate the percentile rank of the value
        n = len(sorted_data)
        percentile_rank = (index / n) * 100

        return percentile_rank

    def check_flow_balance(self, tol=1.00E-05):

        nr_of_vs = self.flownetwork.nr_of_vs
        flow_rate = self.flownetwork.flow_rate
        boundary_vs = self.flownetwork.boundary_vs
        flow_balance = self._get_flow_balance()

        ref_flow = np.abs(flow_rate[boundary_vs[0]])
        tol_flow = tol * ref_flow

        is_inside_node = np.logical_not(np.in1d(np.arange(nr_of_vs), boundary_vs))
        local_balance = np.abs(flow_balance[is_inside_node])
        is_locally_balanced = local_balance < tol_flow
        if False in np.unique(is_locally_balanced):
            import sys
            sys.exit("Is locally balanced: " + str(np.unique(is_locally_balanced)) + "(with tol " + str(tol_flow) + ")")

        balance_boundaries = flow_balance[boundary_vs]
        global_balance = np.abs(np.sum(balance_boundaries))
        is_globally_balanced = global_balance < tol_flow
        if not is_globally_balanced:
            import sys
            sys.exit("Is globally balanced: " + str(is_globally_balanced) + "(with tol " + str(tol_flow) + ")")

        # zero-flow-threshold
        # The zero flow threshold is set as the max of the mass balance error for the internal nodes
        if self.flownetwork.zeroFlowThreshold is None:  # self._PARAMETERS['zeroFlowThreshold'] is True and
            # max of the mass balance error for the internal nodes
            self.flownetwork.zeroFlowThreshold = np.max(local_balance)
            # print to check the value of the threshold
            print("Tolerance :" + str(self.flownetwork.zeroFlowThreshold))

            percentage = self.percentage_lower_than(np.abs(flow_rate), self.flownetwork.zeroFlowThreshold)
            print(f"The percentage of elements lower than {self.flownetwork.zeroFlowThreshold} is: {percentage:.2f}%")

        return
