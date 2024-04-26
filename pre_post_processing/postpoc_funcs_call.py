from postproc_funcs import (create_pressure_histogram, plot_target_reach, analyze_rbc_velocities, calculate_cbf,
                            analyze_pressures)

pressure_path = (r'C:\Master_thesis_2\BCs_tuning_final_network\csv_files'
                 r'\gamma_4000_targets_19_pressures_18329.csv')
nodes_path = r'C:\Master_thesis_2\BCs_tuning_final_network\network\nodes_boundary_data.csv'
final_rbc_values_path = (r'C:\Master_thesis_2\BCs_tuning_final_network\csv_files'
                     r'\gamma_4000_targets_19_rbc_velocities_18329.csv')
forward_model_rbc_path = (r'C:\Master_thesis_2\BCs_tuning_final_network\initial_solution_forward_model'
                      r'\rbc_vels_baseline.csv')
simu_targets_path = r'C:\Master_thesis_2\BCs_tuning_final_network\df_target_edges\baseline_df_mass_balance.csv'
flow_rates_path = r'C:\Master_thesis_2\BCs_tuning_final_network\Postprocessing\csv_files\flow_rate'
pressures_general_path = r'C:\Master_thesis_2\BCs_tuning_final_network\csv_files'
BCs_path = r'C:\Master_thesis_2\BCs_tuning_final_network\csv_files'
edges_data_path = r'C:\master_thesis_2\BCs_tuning_final_network\network\edgesDict_+1.pkl'
nkind_path = r'C:\master_thesis_2\BCs_tuning_final_network\network\nkind.xlsx'

simulation_name = 'baseline_cf_1bb'
simulation_name_cbf = '18329'

# Call the function to create the pressure histogram
create_pressure_histogram(pressure_path, nodes_path, simulation_name)

# Call the function to plot the target reach
plot_target_reach(final_rbc_values_path, forward_model_rbc_path, simu_targets_path, simulation_name)

# Call the function to analyze the red blood cell velocities
analyze_rbc_velocities(final_rbc_values_path, simulation_name)

# Call the function to calculate the CBF
calculate_cbf(flow_rates_path, pressures_general_path, BCs_path, edges_data_path, simulation_name_cbf)

# Call the function to plot the arteries, veins and capillaries pre and post tuning values.
analyze_pressures(nodes_path, pressure_path, nkind_path, simulation_name)

