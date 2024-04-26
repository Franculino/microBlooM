from preproc_funcs import process_and_save_data, create_nkind_file, pressure_initialization

correspondence_path = r'C:\master_thesis_2\network_def\correspondence.xlsx'
base_path = r'C:\master_thesis_2\BCs_tuning_final_network\df_target_edges'
nkind_path = r'C:\master_thesis_2\network_def\nkind.xlsx'
edges_path = r'C:\master_thesis_2\network_+1\edgesDict_+1.pkl'
vertices_path = r'C:\master_thesis_2\network_def\verticesDict.pkl'
cap_pressure_path = r'C:\Master_thesis\final_network/initialisation_BCs.csv'
boundary_data_path = r'C:\master_thesis_2\network_def/nodes_boundary_data_aa.csv'
tuned_parameters_path = r'C:\Master_thesis_2\BCs_tuning_final_network\nodes_tuned/tuned_vertex_parameters.csv'

process_and_save_data(correspondence_path, base_path, 373)
create_nkind_file(correspondence_path, nkind_path, 308)
pressure_initialization(edges_path, vertices_path, cap_pressure_path, nkind_path, boundary_data_path,
                        tuned_parameters_path, 45, 10)
