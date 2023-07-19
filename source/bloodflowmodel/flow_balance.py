import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import source.flow_network as flow_network


class FlowBalance(object):

    def __init__(self, flownetwork: flow_network.FlowNetwork):

        self.flownetwork = flownetwork

    def _get_flow_balance(self):

        nr_of_vs = self.flownetwork.nr_of_vs
        nr_of_es = self.flownetwork.nr_of_es

        edge_list = self.flownetwork.edge_list

        flow_balanceOne = np.zeros(nr_of_vs)
        flow_balanceOneSimple = np.zeros(nr_of_vs)
        flow_balanceOriginal = np.zeros(nr_of_vs)
        flow_rateOne = self.flownetwork.flow_rateOne
        flow_rateOneSimple = self.flownetwork.flow_rateOneSimple
        flow_rateOriginal = self.flownetwork.flow_rateOriginal

        for eid in range(nr_of_es):
            flow_balanceOne[edge_list[eid, 0]] += flow_rateOne[eid]
            flow_balanceOne[edge_list[eid, 1]] -= flow_rateOne[eid]
        for eid in range(nr_of_es):
            flow_balanceOneSimple[edge_list[eid, 0]] += flow_rateOneSimple[eid]
            flow_balanceOneSimple[edge_list[eid, 1]] -= flow_rateOneSimple[eid]
        for eid in range(nr_of_es):
            flow_balanceOriginal[edge_list[eid, 0]] += flow_rateOriginal[eid]
            flow_balanceOriginal[edge_list[eid, 1]] -= flow_rateOriginal[eid]

        return flow_balanceOne, flow_balanceOneSimple, flow_balanceOriginal

    def check_flow_balance(self):

        nr_of_vs = self.flownetwork.nr_of_vs
        flow_rateOne = self.flownetwork.flow_rateOne
        flow_rateOneSimple = self.flownetwork.flow_rateOneSimple
        flow_rateOriginal = self.flownetwork.flow_rateOriginal
        boundary_vs = self.flownetwork.boundary_vs

        flow_balanceOne, flow_balanceOneSimple, flow_balanceOriginal = self._get_flow_balance()

        # Original
        ref_flowOriginal = np.abs(flow_rateOriginal[boundary_vs[0]])
        is_inside_nodeOriginal = np.logical_not(np.in1d(np.arange(nr_of_vs), boundary_vs))
        local_balanceOriginal = np.abs(flow_balanceOriginal[is_inside_nodeOriginal])

        balance_boundariesOriginal = flow_balanceOriginal[boundary_vs]
        global_balanceOriginal = np.abs(balance_boundariesOriginal)
        self.frequency_plot(local_balanceOriginal, "Mass Balance Error of Original Approach for internal node", "Mass Balance Error", '#a0c4ff')

        # One
        ref_flowOne = np.abs(flow_rateOne[boundary_vs[0]])
        is_inside_nodeOne = np.logical_not(np.in1d(np.arange(nr_of_vs), boundary_vs))
        local_balanceOne = np.abs(flow_balanceOne[is_inside_nodeOne])
        self.frequency_plot(local_balanceOne, "Mass Balance Error of No One for internal node", "Mass Balance Error", '#84dcc6')

        balance_boundariesOne = flow_balanceOne[boundary_vs]
        global_balanceOne = np.abs(balance_boundariesOne)

        # OneSimple
        ref_flowOneSimple = np.abs(flow_rateOneSimple[boundary_vs[0]])
        is_inside_nodeOneSimple = np.logical_not(np.in1d(np.arange(nr_of_vs), boundary_vs))
        local_balanceOneSimple = np.abs(flow_balanceOneSimple[is_inside_nodeOneSimple])
        self.frequency_plot(local_balanceOneSimple, "Mass Balance Error of No One Simple for internal node", "Mass Balance Error", '#ffa69e')

        balance_boundariesOneSimple = flow_balanceOneSimple[boundary_vs]
        global_balanceOneSimple = np.abs(balance_boundariesOneSimple)

        return sys.exit()

    def frequency_plot(self, data, title, x_axis, color_plot):

        # Calculate statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        max_val = np.max(data)
        print(np.min(data[data != 0]))
        print("The value for " + str(title) + " of: median = " + str(median_val) + " mean = " + str(mean_val) + " max = " + str(max_val))

        # Create the histogram with seaborn
        bin_count = 20000
        sns.histplot(data, bins=bin_count, kde=False, color=color_plot, edgecolor='white', stat="percent")
        # Customize the plot with matplotlib
        plt.xlabel(x_axis)
        plt.ylabel('Percentage (%)')  # Update y-axis label
        plt.title(title)
        sns.despine(left=True)  # Remove the left and top spines
        plt.grid(axis='y', alpha=0.5)  # Add gridlines to the y-axis
        plt.xticks(fontsize=8)  # Adjust x-axis tick font size

        plt.tick_params(axis='y', which='both', color='#f0f0f0')

        # Calculate bin heights (frequencies) and total data points
        bin_heights, _ = np.histogram(data, bins=bin_count)
        total_points = len(data)

        # Calculate the percentage of elements in each bin and set as y-axis tick labels
        # percentage_labels = [f'{(height / total_points) * 100:.2f}%' for height in bin_heights]

        # Add vertical lines for mean, median, and max
        plt.axvline(mean_val, color='red', linestyle='dashed', label='Mean', linewidth=1)
        plt.axvline(median_val, color='blue', linestyle='dashed', label='Median', linewidth=1)
        plt.axvline(max_val, color='green', linestyle='dashed', label='Max', linewidth=1)

        # Log y-axis
        plt.xscale('log')

        # Add a legend
        plt.legend(loc='upper right')
        plt.ylim(0, 100)
        # Set y-axis tick locations and labels

        # Add a background color to the plot area
        plt.gca().set_facecolor('#f0f0f0')
        # plt.savefig("/Users/cucciolo/Desktop/microBlooM/data_plot/" + str(title) + ".png", dpi=300)
        # Display the plot
        plt.tight_layout()
        plt.show()
        # plt.close()

    def check_flow_balance_Original(self, tol=1.00E-05):

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

        return
