import os
import pickle
import igraph as ig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_pressure_histogram(pressure_path, nodes_path, simulation_name):
    """
    Creates a histogram of pressures at boundary nodes from simulation data.

    This function reads pressure values from a CSV file specified by pressure_path,
    selects the pressures at boundary nodes defined in the CSV file specified by nodes_path,
    and plots a histogram of these pressures. The plot is saved as a PNG file with a name
    based on the simulation_name parameter.

    Parameters:
    pressure_path (str): Path to the CSV file containing pressure data.
    nodes_path (str): Path to the CSV file containing boundary node IDs.
    simulation_name (str): A string to be included in the name of the saved figure file.
    """
    pressures_array = pd.read_csv(pressure_path)['pressures'].to_numpy()
    boundary_nodes = pd.read_csv(nodes_path)['nodeID'].to_numpy()

    boundary_pressures = pressures_array[boundary_nodes] / 133.322

    # Create the plot
    plt.figure(figsize=(6, 5))

    # Plot the histogram
    plt.hist(boundary_pressures, bins=50, color='blue', edgecolor='black')
    plt.title(f'{simulation_name}')
    plt.xlabel('Pressure [mmHg]')
    plt.ylabel('Frequency')

    # Adjust the layout of the plot
    plt.tight_layout()

    save_path = (r'C:\Master_thesis_2\BCs_tuning_final_network\Postprocessing\png_files'
                 rf'\boundary_pressure_histogram_{simulation_name}.png')
    plt.savefig(save_path, dpi=600)
    plt.close()  # Close the figure to free memory


def plot_target_reach(final_values_path, forward_model_path, simu_targets_path, simulation_name):
    """
    Plots the target reach analysis comparing pre-tuning and post-tuning red blood cell (RBC) velocities
    against exact target values.
    The function reads data from specified paths and generates a scatter plot with additional
    information on target accuracy.

    Parameters:
    final_values_path (str): Path to the CSV file containing final RBC velocities.
    forward_model_path (str): Path to the CSV file containing initial RBC velocities from the forward model.
    simu_targets_path (str): Path to the CSV file containing target edge values and ids.
    simulation_name (str): Name of the simulation, used in the output image file name.
    """
    simu_targets_df = pd.read_csv(simu_targets_path)
    targets = simu_targets_df['edge_tar_eid'].to_numpy()
    target_values = simu_targets_df['edge_tar_value'].to_numpy() * 1000

    forward_values = pd.read_csv(forward_model_path)['rbc_velocities'].to_numpy() * 1000
    final_values = pd.read_csv(final_values_path)['rbc_velocities'].to_numpy() * 1000

    error = np.abs(target_values - final_values[targets]) * 100 / np.abs(final_values[targets])
    print("Target value error: ", error)

    target_forward_values = forward_values[targets]
    target_final_values = final_values[targets]

    n_targets = np.arange(len(targets))  # Array from 0 to len(targets)

    # Create scatter plot
    plt.scatter(n_targets, target_forward_values, marker='x', color='red', label='Pre-tuning')
    plt.scatter(n_targets, target_final_values, color='red', label='Post-tuning', s=5)

    # Add shaded squares
    square_size = 0.2 * target_values
    for i, (x, y, z, size, name) in enumerate(
            zip(n_targets, target_values, target_forward_values, square_size, targets)):
        rect = plt.Rectangle((x - size / 2, y - size / 2), size, size, color='gray', alpha=0.5,
                             label='+-10% range' if i == 0 else None)
        plt.plot(x, y, marker='_', markersize=3, color='blue', label='Exact target value' if i == 0 else None)
        plt.gca().add_patch(rect)
        plt.text(x, z - 0.5, str(name), ha='center', va='center', color='black')

    # Customize the plot
    plt.title('Target value pre-tuning and post-tuning')
    plt.xlabel('Number of target values')
    plt.ylabel('RBC velocity [mm/s]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(False)
    plt.xticks(np.arange(0, len(targets), 1))

    # Save the plot
    save_path = (
        rf'C:\Master_thesis_2\BCs_tuning_final_network\Postprocessing\png_files\targets_reached_{simulation_name}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()  # Close the plot to free memory


def analyze_rbc_velocities(rbc_path, simulation_name):
    """
    Analyzes the red blood cell (RBC) velocities from a CSV file, calculates statistical data,
    and generates histograms of the RBC velocities and their physiological range.

    Parameters:
    rbc_path (str): Path to the CSV file containing RBC velocities data.
    simulation_name (str): Name of the simulation, used in the output image file name.
    """
    # Read the CSV file and convert the DataFrame to a NumPy array
    rbc = np.abs(pd.read_csv(rbc_path)['rbc_velocities'].to_numpy()).flatten() * 1000

    # Calculate the range of the data
    data_range = np.max(rbc) - np.min(rbc)

    # Calculate the number of bins for the histogram
    num_bins = int(data_range / 0.1)

    # Calculate the median and standard deviation
    median = np.median(rbc)
    std = np.std(rbc)

    # Count the elements outside of the specified range
    out_of_range_count = np.sum((rbc > 20) | (rbc < 0.05))
    percentage_out_range = out_of_range_count / len(rbc) * 100

    # Print the results
    print("Median:", median)
    print("Standard deviation:", std)
    print("Number of blood vessels out of range:", out_of_range_count)
    print("Percentage out of range:", percentage_out_range)

    # Create a histogram
    plt.hist(rbc, bins=200, range=(np.min(rbc), np.max(rbc)), color='blue', alpha=0.7)
    plt.xlabel('RBC Velocities (mm/s)')
    plt.ylabel('Frequency')
    plt.title('RBC velocity histogram')
    plt.legend()

    save_path_histogram = (f'C:\\Master_thesis_2\\BCs_tuning_final_network\\Postprocessing\\png_files'
                           f'\\histogram_{simulation_name}.png')
    plt.savefig(save_path_histogram)
    plt.close()

    # Bar chart for physiological range
    below_005 = np.sum(rbc < 0.05)
    between_005_and_10 = np.sum((rbc >= 0.05) & (rbc <= 20))
    above_10 = np.sum(rbc > 20)

    categories = ['<0.05', '0.05-20', '>20']
    values = [below_005, between_005_and_10, above_10]

    plt.bar(categories, values, color=['blue', 'green', 'red'], alpha=0.7)
    plt.xlabel('RBC velocity [mm/s]')
    plt.ylabel('Frequency')
    plt.title('RBC Velocity Distribution')

    # Add text labels above the bars
    for bar in plt.bar(categories, values, color=['blue', 'green', 'red'], alpha=0.7):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, round(height, 2), ha='center', va='bottom')

    save_path_bar = (f'C:\\Master_thesis_2\\BCs_tuning_final_network\\Postprocessing\\png_files'
                     f'\\physiological_range_{simulation_name}.png')
    plt.savefig(save_path_bar)
    plt.close()


def calculate_cbf(flow_rates_path, pressures_path, BCs_path, edges_data_path, simulation):
    """
        This function calculates the cerebral blood flow (CBF) for a given set of simulations. It processes data from
        specified paths, computes the inflow rates based on boundary conditions and pressures, and then calculates the
        CBF based on these inflows.

        Parameters:
        flow_rates_path (str): Path to the directory containing flow rate CSV files.
        pressures_path (str): Path to the directory containing pressure CSV files.
        BCs_path (str): Path to the directory containing boundary condition pressure CSV files.
        edges_data_path (str): Path to the pickle file containing the network's edges data.
        simulation_name (str): Name of the simulation. If set to 'baseline', the function will calculate the CBF for
        all baseline simulations.

        The function loads the network graph, reads the flow, pressure, and boundary condition files, filters them
        based on the simulation name, and calculates the total inflow for each simulation. The CBF is then calculated
        based on the total inflow and the volume and density of the brain region simulated. The results are printed
        out for each file processed.
    """
    with open(edges_data_path, 'rb') as file:
        # Load data from the .pkl file using the appropriate protocol
        edges_data = pickle.load(file, encoding='latin1')

    # Convert tuple to adjacency list and create graph
    adjlist = np.array(edges_data['connectivity'])
    graph = ig.Graph(adjlist.tolist())
    ig.summary(graph)

    # Identify boundary condition vertices
    bc_vertices = np.where(np.array(graph.degree()) == 1)[0]
    neighbors = np.ravel(list(map(lambda bc: list(graph.neighbors(bc)), bc_vertices)))

    # List files related to flow rates, pressures, and boundary conditions
    flow_rate_files = [f for f in os.listdir(flow_rates_path) if simulation in f]
    pressures_files = [f for f in os.listdir(pressures_path) if '_pressures_' in f and simulation in f]
    BCs_pressures_files = [f for f in os.listdir(BCs_path) if 'bcs_BCs_pressure' in f and simulation in f]

    # Initialize array to sum the inflows
    inflows = np.zeros(len(flow_rate_files))

    for i, (flow_file, pressure_file, BCs_pressure_file) in enumerate(zip(flow_rate_files, pressures_files, BCs_pressures_files)):
        # Read flow rate and pressure files
        flow_rate = pd.read_csv(os.path.join(flow_rates_path, flow_file))["flow_rate"].to_numpy()
        pressure = pd.read_csv(os.path.join(pressures_path, pressure_file))["pressures"].to_numpy()
        BCs_pressure = pd.read_csv(os.path.join(BCs_path, BCs_pressure_file), header=None).iloc[-1, 1:].to_numpy()

        # Create mask for the condition
        condition_mask_cte_BCs = BCs_pressure > pressure[neighbors]
        edge_ids_cte_BCs = np.ravel(list(map(lambda BC, neighbor: graph.get_eid(BC, neighbor)
                                             , bc_vertices, neighbors)))

        # Calculate conditional sum of inflows
        inflow_cte_BCs = np.sum(abs(flow_rate[edge_ids_cte_BCs[condition_mask_cte_BCs]]))
        inflows[i] = inflow_cte_BCs

        # Calculate CBF
        x_dir, y_dir, z_dir = 0.001, 0.00124348, 0.0013  # [m]
        rho = 1047  # [kg/m^3]
        volume = x_dir * y_dir * z_dir  # [m^3]
        grammes = volume * rho * 1000  # [g]
        cbf = inflows * 60 / (1e-6 * grammes)  # [ml/min g]

        print(f"File: {flow_file}, CBF: {cbf[i]}")


def analyze_pressures(nodes_path, pressures_path, nkind_path, simulation_name):
    """
    This function analyzes pressure data from boundary nodes in a vascular network simulation.
    It reads node and pressure data from CSV files, categorizes nodes into arteries, veins, and capillaries,
    and plots initial and final pressure values for each type, saving the plots to disk.

    Args:
    - nodes_path (str): Path to the CSV file containing boundary node data.
    - pressures_path (str): Path to the CSV file containing pressure data.
    - nkind_path (str): Path to the CSV file with node types (arteries, veins, capillaries).
    - simulation_name (str): Name of the simulation for labeling and file naming purposes.
    """
    boundary_nodes_df = pd.read_csv(nodes_path)
    boundary_nodes = boundary_nodes_df['nodeID'].to_numpy()
    pressures_array = pd.read_csv(pressures_path)['pressures'].to_numpy() / 133.322
    nkind_df = pd.read_excel(nkind_path)

    nodeIDs_artery_boundaries = nkind_df.loc[
        nkind_df['nodeID'].isin(boundary_nodes) & (nkind_df['nkind'] == 2), 'nodeID'].to_numpy()
    nodeIDs_vein_boundaries = nkind_df.loc[
        nkind_df['nodeID'].isin(boundary_nodes) & (nkind_df['nkind'] == 3), 'nodeID'].to_numpy()
    nodeIDs_caps_boundaries = nkind_df.loc[
        nkind_df['nodeID'].isin(boundary_nodes) & (nkind_df['nkind'] == 4), 'nodeID'].to_numpy()

    artery_init_value = np.ones(len(nodeIDs_artery_boundaries)) * 47
    vein_init_value = np.ones(len(nodeIDs_vein_boundaries)) * 12
    caps_init_value = (boundary_nodes_df.loc[
                           boundary_nodes_df['nodeID'].isin(nodeIDs_caps_boundaries), 'p'].to_numpy() / 133.322)

    artery_final_value = pressures_array[nodeIDs_artery_boundaries].flatten()
    vein_final_value = pressures_array[nodeIDs_vein_boundaries].flatten()
    caps_final_value = pressures_array[nodeIDs_caps_boundaries].flatten()

    # Plotting and saving for arteries
    plt.scatter(np.arange(len(nodeIDs_artery_boundaries)), artery_init_value, marker='x', color='red',
                label='Pre-tuning')
    plt.scatter(np.arange(len(nodeIDs_artery_boundaries)), artery_final_value, color='green',
                label='Post-tuning')
    plt.title(f'Boundary Arteries Pressure {simulation_name}')
    plt.xlabel('Number of boundary arteries')
    plt.ylabel('Pressure [mmHg]')
    plt.legend()
    plt.grid(False)
    plt.xticks(np.arange(0, len(nodeIDs_artery_boundaries), 1))
    plt.savefig(
        f'C:\\Master_thesis_2\\BCs_tuning_final_network\\Postprocessing\\png_files'
        f'\\arteries_{simulation_name}.png',
        dpi=600)
    plt.close()

    # Plotting and saving for veins
    plt.scatter(np.arange(len(nodeIDs_vein_boundaries)), vein_init_value, marker='x', color='red',
                label='Pre-tuning')
    plt.scatter(np.arange(len(nodeIDs_vein_boundaries)), vein_final_value, color='green', label='Post-tuning')
    plt.title(f'Boundary Veins Pressure {simulation_name}')
    plt.xlabel('Number of boundary veins')
    plt.ylabel('Pressure [mmHg]')
    plt.legend()
    plt.grid(False)
    plt.xticks(np.arange(0, len(nodeIDs_vein_boundaries), 1))
    plt.savefig(
        f'C:\\Master_thesis_2\\BCs_tuning_final_network\\Postprocessing\\png_files'
        f'\\veins_{simulation_name}.png',
        dpi=600)
    plt.close()

    # Plotting and saving for capillaries
    fig, ax = plt.subplots(figsize=(18, 6))
    plt.scatter(np.arange(len(nodeIDs_caps_boundaries)), caps_init_value, marker='x', color='red',
                label='Pre-tuning', s=17)
    plt.scatter(np.arange(len(nodeIDs_caps_boundaries)), caps_final_value, color='green', label='Post-tuning', s=17)
    plt.title(f'Boundary Capillaries Pressure {simulation_name}')
    plt.xlabel('Boundary index')
    plt.ylabel('Pressure [mmHg]')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(len(nodeIDs_caps_boundaries)), nodeIDs_caps_boundaries, rotation=90, ha="right",
               fontsize=10)
    plt.subplots_adjust(wspace=5)
    plt.savefig(
        f'C:\\Master_thesis_2\\BCs_tuning_final_network\\Postprocessing\\png_files'
        f'\\capillaries_{simulation_name}.png',
        dpi=600)
    plt.close()


