import pickle
import numpy as np
import pandas as pd


def process_and_save_data(correspondence_path, base_path, n_edges):
    """
    This function processes the experimental measurements data from an Excel file, creates target datasets for different
    simulation states, and appends additional target data to all datasets. It saves all data in CSV format.

    Args:
    - correspondence_path (str): Path to the Excel file containing measurement correspondences.
    - base_path (str): Base directory path where the CSV files will be saved.
    - n_edges (int): number of edges of the network.
    """

    # Read data from Excel
    df = pd.read_excel(correspondence_path)
    df = df.query("`Edge index` != '-'")

    targets = df['Edge index'].to_numpy().astype(int)
    baseline_vels = df['Baseline vRBC (mm/s)'].to_numpy() * 0.001
    stroke_vels = df['Post vRBC'].to_numpy() * 0.001
    stroke_2h_vels = df['Post 2h vRBC'].to_numpy() * 0.001
    stroke_24h_vels = df['Post 24h vRBC'].to_numpy() * 0.001

    data_frames = {
        'baseline_df': baseline_vels,
        'stroke_df': stroke_vels,
        'stroke_2h_df': stroke_2h_vels,
        'stroke_24h_df': stroke_24h_vels
    }

    all_updated_dfs = {}

    for key, values in data_frames.items():
        # Create initial DataFrame from the data
        initial_df = pd.DataFrame({
            'edge_tar_eid': targets,
            'edge_tar_type': 2,
            'edge_tar_value': values,
            'edge_tar_range_pm': 0,
            'edge_tar_sigma': values
        })
        initial_df.to_csv(f'{base_path}\\{key}.csv', index=False, sep=',')

        # Add ranges to the DataFrame
        ranges = np.setdiff1d(np.arange(n_edges), targets)
        sigma = 10.025e-3
        new_rows_df = pd.DataFrame({
            'edge_tar_eid': ranges,
            'edge_tar_type': 2,
            'edge_tar_value': sigma,
            'edge_tar_range_pm': sigma - 0.05e-3,
            'edge_tar_sigma': sigma
        })

        updated_df = pd.concat([initial_df, new_rows_df])
        updated_df = updated_df.sort_values(by='edge_tar_eid').reset_index(drop=True)

        # Save updated DataFrame to CSV
        updated_df.to_csv(f'{base_path}\\{key}_ranges_005_20.csv', index=False, sep=',')
        all_updated_dfs[key] = updated_df


def create_nkind_file(correspondence_path, output_path, n_nodes):
    """
    Reads node vessel type data from a specified Excel sheet and writes processed node types to a new Excel file.

    Args:
    - correspondence_path (str): Path to the Excel file containing the node data.
    - output_path (str): Path where the output Excel file will be saved.
    """
    sheet_number = 2  # Sheet index starts from 0, so 2 refers to the third sheet
    df = pd.read_excel(correspondence_path, sheet_name=sheet_number)

    labelling = df['Labelling (vertex)'].to_numpy()
    node_type = df['Type'].to_numpy()

    # Remove NaN values from labelling and corresponding values in node_type
    valid_indices = ~np.isnan(labelling)
    labelling = labelling[valid_indices].astype(int)  # Ensure labelling is of integer type for indexing
    node_type = node_type[valid_indices]

    # Fill NaN values with the previous non-NaN value without a loop
    type_filled = np.array(node_type)
    mask = np.isnan(type_filled)
    indices = np.where(~mask, np.arange(len(type_filled)), 0)
    np.maximum.accumulate(indices, out=indices)
    type_filled = type_filled[indices]

    # Initialize nkind with 4 (capillary)
    nkind = np.full(n_nodes, 4)
    nkind[labelling] = type_filled  # Change nkind values at labelling indices to the corresponding node_type values

    # Create the DataFrame
    df_nkind = pd.DataFrame({
        'Node index': np.arange(1, n_nodes + 1),  # Assuming nodes are indexed from 1 to n_nodes
        'Nkind': nkind
    })

    # Save the DataFrame to Excel
    df_nkind.to_excel(output_path, index=False)


def pressure_initialization(edges_path, vertices_path, cap_pressure_path, nkind_path, boundary_data_path,
                            tuned_parameters_path, artery_pressure, vein_pressure):
    """
    Processes edge and vertex data from pickle files, calculates boundary conditions based on node types,
    and saves the boundary conditions and parameters to be tuned for capillaries into CSV files.

    Args:
    - edges_path (str): Path to the edges data pickle file.
    - vertices_path (str): Path to the vertices data pickle file.
    - cap_pressure_path (str): Path to the capillary pressures CSV file.
    - nkind_path (str): Path to the Excel file containing node kinds.
    - boundary_data_path (str): Output path for the nodes boundary data CSV.
    - tuned_parameters_path (str): Output path for the tuned vertex parameters CSV.
    - artery_pressure: pressure arteries have to be initialized with
    - vein_pressure: pressure veins have to be initialized with
    """
    # Load data from pickle files
    with open(edges_path, 'rb') as file:
        edges_data = pickle.load(file, encoding='latin1')
    with open(vertices_path, 'rb') as file:
        vertices_data = pickle.load(file, encoding='latin1')

    # Load additional data
    cap_pressure_df = pd.read_csv(cap_pressure_path)
    nkind_df = pd.read_excel(nkind_path)

    # Extract and preprocess relevant data
    connectivity = np.concatenate(np.array(edges_data['connectivity']))
    unique_elements, counts = np.unique(connectivity, return_counts=True)
    boundary_vs = unique_elements[counts == 1]

    # Initialize pressure array for boundary nodes
    pressure_array = np.zeros_like(boundary_vs, dtype=float)
    nodeIDs_artery = nkind_df.loc[nkind_df['Nkind'] == 2, 'Node index'].to_numpy()
    nodeIDs_veins = nkind_df.loc[nkind_df['Nkind'] == 3, 'Node index'].to_numpy()
    nodeIDs_caps = nkind_df.loc[nkind_df['Nkind'] == 4, 'Node index'].to_numpy()

    # Assign pressures based on node types
    artery_pressure *= 133.322
    vein_pressure *= 133.322
    pressure_array[np.isin(boundary_vs, nodeIDs_artery)] = artery_pressure
    pressure_array[np.isin(boundary_vs, nodeIDs_veins)] = vein_pressure

    # Calculate capillary pressures based on depth
    depth_all = np.array(vertices_data['coords'])[:, 0]
    median_caps = cap_pressure_df['Median Pressure capillaries'].to_numpy()
    std_caps = cap_pressure_df['Std Deviation capillaries'].to_numpy() / 2
    pressure_caps = np.random.uniform(median_caps - std_caps, median_caps + std_caps) * 133.322

    depth_ranges = cap_pressure_df['Depth range'].to_numpy()
    # Convertir las cadenas a tuplas usando eval
    depth_ranges = [eval(rng) for rng in depth_ranges]
    depth_ranges_array = np.array(depth_ranges).flatten()
    depth_ranges_array = np.unique(depth_ranges_array)

    # Determine depth and assign pressures
    boundaries_caps_IDs_boundaries_order = boundary_vs[np.isin(boundary_vs, nodeIDs_caps)]
    depth_caps = depth_all[boundaries_caps_IDs_boundaries_order]
    depth_indices = np.digitize(depth_caps, depth_ranges_array, right=True)
    pressure_array[np.isin(boundary_vs, nodeIDs_caps)] = pressure_caps[depth_indices - 1]

    # Save boundary data to CSV
    boundary_data_df = pd.DataFrame({'nodeID': boundary_vs, 'boundaryType': 1, 'p': pressure_array})
    boundary_data_df.to_csv(boundary_data_path, index=False, sep=',')

    # Save parameters to be tuned for capillaries to CSV
    tuned_vertex_parameters_df = pd.DataFrame({
        'vertex_param_vid': boundaries_caps_IDs_boundaries_order,
        'vertex_param_pm_range': np.ones_like(boundaries_caps_IDs_boundaries_order)
    })
    tuned_vertex_parameters_df.to_csv(tuned_parameters_path, index=False)





