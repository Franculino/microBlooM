"""
A python script to estimate edge parameters such as diameters and transmissibilities of microvascular networks based
on given flow rates and velocities in selected edges. Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Update the boundary pressures with a gradient descent algorithm minimising a given cost function.
5. Restriction of parameters to desired ranges (target value +/- tolerance).
6. Individual selection of parameter boundary vertices and target edges.
7. Target flow rates and velocities can be specified and combined into a single cost function.
8. Tuning of absolute boundary pressures.
9. Optimisation of pressures for a fixed number of iteration steps.
10. Save the results in a file.
"""
import sys

import numpy as np

from source.flow_network import FlowNetwork
from source.inverse_model import InverseModel
from types import MappingProxyType
from source.exportdata.BCs_monitoring import BCs_monitoring
from source.exportdata.solution_monitoring_inverseproblem import SolutionMonitoring
import source.setup.setup as setup


# MappingProxyType is basically a const dict
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 5,  # 1: generate hexagonal graph
                                   # 2: import graph from csv files
                                   # 3: import graph fromgra iph file (pickle file)
                                   # 4: todo import graph from edge_data and vertex_data pickle files
                                   # 5: import tortuous graph
        # True it writes a tortuous network, False all the vessels are straight (only visualization)
        "tortuous": False,  # Only can be True when read_network_option is 5 and write_network_option 3.
        "write_network_option": 3,  # 1: do not write anything
                                    # 2: igraph format
                                    # 3: write to vtp format
                                    # 4-...: todo other file formats. also handle overwrite data, etc
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
                                       # 3: todo: RBC tracking
                                       # 4-xxx: todo: steady state RBC laws
        "rbc_impact_option": 3,  # 1: hd = ht (makes only sense if tube_haematocrit_option:1 or ht=0)
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: todo Other laws. in vivo?
        "solver_option": 1,  # 1: Direct solver
                             # 2: todo: other solvers (CG, AMG, ...)
        # Blood properties
        "ht_constant": 0.3,
        "mu_plasma": 0.0012,
        # Hexagonal network properties
        "nr_of_hexagon_x": 5,
        "nr_of_hexagon_y": 5,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 65, 21, 48, 41, 31, 56],
        "hexa_boundary_values": [2, 1, 3, 1, 1, 5, 2],
        "hexa_boundary_types": [1, 1, 1, 1, 1, 1, 1],  # 1: pressure, 2: flow rate
        # Import network from csv options
        "csv_path_vertex_data": r'C:\Master_thesis_2\BCs_tuning_final_network\network/node_data.csv',
        "csv_path_edge_data": r'C:\Master_thesis_2\BCs_tuning_final_network\network/edge_data.csv',
        "csv_path_boundary_data": r'C:\Master_thesis_2\BCs_tuning_final_network\network'
                                  r'/nodes_boundary_data.csv',
        "csv_diameter": "d", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeID", "csv_boundary_type": "boundaryType", "csv_boundary_value": "p",

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": r'C:\Master_thesis_2\Modelo-Artorg\trial-capillary-network\igraph_pickle.pkl',
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_velocities": 'v',
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Import pkl files for reading tortuous graph
        "edges_pkl": r'C:\master_thesis_2\network_+1\edgesDict_+1.pkl',
        "vertices_pkl": r'C:\master_thesis_2\network_+1\verticesDict_+1.pkl',
        "boundary_data": r'C:\master_thesis_2\BCs_tuning_final_network\network'
                         r'\nodes_boundary_data_1.csv',

        # Write options
        "write_override_initial_graph": False,
        "write_path_igraph": r'C:\Master_thesis_2\BCs_tuning_final_network\vtp_files/baseline_cf_2_8bb.vtp',  # only required for "write_network_option" 2

        # Inverse problem options
        # Inverse problem options
        "cost_function_option": 1,  # 1: Cost function only with targets term
                                    # 2: Two terms, target values and ranges term in absolute value
                                    # 3: Two terms, target values but with a % of error (treated as ranges) and ranges
                                    # term in absolute value
        # Define parameter space
        "parameter_space": 11,  # 1: Relative diameter to baseline (alpha = d/d_base)
                                # 2: Relative transmissibility to baseline (alpha = T/T_base)
                                # 11: Pressure boundary condition values (alpha = p_0)
        "parameter_restriction": 1,  # 1: No restriction of parameter values (alpha_prime = alpha)
                                     # 2: Restriction of parameter by a +/- tolerance to baseline
        "inverse_model_solver": 1,  # Direct solver
                                    # 2: todo: other solvers (CG, AMG, ...)
        # Filepath to prescribe target values / constraints on edges
        "csv_path_edge_target_data": r'C:\master_thesis_2\BCs_tuning_final_network\df_target_edges'
                                     r'/baseline_df_mass_balance.csv',
        # CSV file with only the measurement values in order to plot them
        "csv_path_edge_target_measurements": r'C:\master_thesis_2\BCs_tuning_final_network\df_target_edges'
                                             r'/baseline_df_mass_balance.csv',
        # Filepath to define the edge parameter space (only for tuning of diameters and transmissibilities)
        "csv_path_edge_parameterspace": "not needed",
        # Filepath to define the vertex parameter space (only for tuning of boundary conditions)
        "csv_path_vertex_parameterspace": r'C:\Master_thesis_2\BCs_tuning_final_network\nodes_tuned'
                                          r'/nodes_tuned_data_p_change.csv',
        # Gradient descent options:
        "gamma": 4000,  # parameters change gamma
        "variable_gamma": True,  # True if gamma increases its value every "its_gamma" iterations
        "its_gamma": 2000,  # Gamma increases its value every "its_gamma" iterations
        "increased_gamma": 2000,  # the value gamma is increased every "its_gamma" iterations
        "phi": .5,  # for parameter_restriction 2
        "max_nr_of_iterations": 1000,  # Maximum of iterations
        "n_targets": 19,  # Number of velocities chosen in trials
        # Output
        "csv_path_solution_monitoring": r'C:\Master_thesis_2\BCs_tuning_final_network\csv_files/',
        "png_path_solution_monitoring": r'C:\Master_thesis_2\BCs_tuning_final_network\png_files/',
        # Threshold to stop the simulation
        "threshold": 0.02,
        "ranges_weight": 0.3,
    }
)

setup_simulation = setup.SetupSimulation()

# Initialise objects related to simulate blood flow without RBC tracking.
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
        imp_solver = setup_simulation.setup_bloodflow_model(PARAMETERS)

# Initialise objects related to the inverse model.
imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter, imp_adjoint_solver, \
    imp_alpha_mapping = setup_simulation.setup_inverse_model(PARAMETERS)

# Initialise flownetwork and inverse model objects
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)

inverse_model = InverseModel(flow_network, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                             imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)
bcs_monitoring = BCs_monitoring(flow_network, inverse_model, PARAMETERS)
solution_monitoring = SolutionMonitoring(flow_network, inverse_model, PARAMETERS)

print("Read network: ...")
flow_network.read_network()
print("Read network: DONE")

print("Update transmissibility: ...")
flow_network.update_transmissibility()
print("Update transmissibility: DONE")

print("Update flow, pressure and velocity: ...")
flow_network.update_blood_flow()
print("Update flow, pressure and velocity: DONE")

inverse_model.initialise_inverse_model()
inverse_model.update_cost()

nr_of_iterations = int(PARAMETERS["max_nr_of_iterations"])
print("Solve the inverse problem and update the diameters: ...")

i = 0
# First iteration out of the loop to avoid problems with the value of current_target_values. Otherwise, it would be None
inverse_model.current_iteration = int(i+1)
inverse_model.update_state()
flow_network.update_transmissibility()
flow_network.update_blood_flow()
inverse_model.update_cost()
inverse_model.update_target_values()

# Calling the function to get the new Boundary conditions in each iteration
iteration_array = bcs_monitoring.get_arrays()
# Calling the function to monitor the solution
(n_target, n_range) = solution_monitoring.get_arrays_for_plots()
param_dict = dict(PARAMETERS)
param_dict["gamma"] = PARAMETERS["gamma"]
PARAMETERS = MappingProxyType(param_dict)
dif = np.abs(inverse_model.current_measurement_values - inverse_model.measurements_value)
reached = 0
# for i in range(nr_of_iterations):
while reached < 19:
    f_h_previous = inverse_model.f_h
    i += 1
    inverse_model.current_iteration = int(i+1)
    inverse_model.update_state()
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    inverse_model.update_cost()
    inverse_model.update_target_values()
    # Calling the function to get the new Boundary conditions in each iteration
    iteration_array = bcs_monitoring.get_arrays()
    # Calling the function to monitor the solution
    (n_target, n_range) = solution_monitoring.get_arrays_for_plots()
    dif = np.abs(inverse_model.current_measurement_values - inverse_model.measurements_value)
    reached = np.sum(dif < abs(PARAMETERS["threshold"] * inverse_model.measurements_value))

    if i % 100 == 0:
        print(str(i)+" / " + str(nr_of_iterations) + " iterations done (f_H =", "%.2e" % inverse_model.f_h+")")
        print("Cost target: ", inverse_model.f_h_target, "Cost range: ", inverse_model.f_h_range)
        print(f"Reached {reached} values out of", PARAMETERS["n_targets"])

    if i % PARAMETERS["its_gamma"] == 0 and f_h_previous < inverse_model.f_h:
        param_dict = dict(PARAMETERS)
        param_dict["gamma"] += PARAMETERS["increased_gamma"]
        PARAMETERS = MappingProxyType(param_dict)
        inverse_model.update_gamma(PARAMETERS)

    # In case some monitoring is done during the simulation
    if i % 2000 == 0:
        bcs_monitoring.plot_BCs_vs_iterations(iteration_array)
        solution_monitoring.plot_cost_function_vs_iterations()
        flow_network.write_network()

print(str(i)+" / " + str(nr_of_iterations) + " iterations done (f_H =", "%.2e" % inverse_model.f_h+")")
print("Cost target: ", inverse_model.f_h_target, "Cost range: ", inverse_model.f_h_range)
print("Solve the inverse problem and update the diameters: DONE")

# Save csv files and figures
print("Creating and saving plots and csv files: ...")
bcs_monitoring.BCs_csv()
bcs_monitoring.plot_BCs_vs_iterations(iteration_array)
solution_monitoring.plot_cost_function_vs_iterations()
solution_monitoring.cost_function_csv()
solution_monitoring.flow_rate_csv()
solution_monitoring.pressures_csv()
solution_monitoring.rbc_velocity_csv()
if n_target > 0:
    solution_monitoring.target_edges_csv()
    solution_monitoring.target_edges_plot(n_target)
if n_range > 0:
    solution_monitoring.range_edges_csv()

print("Creating and saving plots and csv files: DONE")

# Write the results to file
flow_network.write_network()
