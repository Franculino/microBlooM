import sys
import time
import numpy as np
import pandas as pd
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib.pyplot as plt

from source.flow_network import FlowNetwork
from source.bloodflowmodel.flow_balance import FlowBalance
from types import MappingProxyType
from source.particle_tracking.passive_particle_tracking_v2 import Particle_tracker
import source.setup.setup as setup

# MappingProxyType is basically a const dict.
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 3,  # 1: generate hexagonal graph
                                   # 2: import graph from csv files
                                   # 3: import graph from igraph format (pickle file)
        "write_network_option": 1,  # 1: do not write anything
                                    # 2: write to igraph format (.pkl)
                                    # 3: write to vtp format (.vtp)
                                    # 4: write to two csv files (.csv)
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
        "rbc_impact_option": 3,  # 1: No RBCs (hd=0) - makes only sense if tube_haematocrit_option:1 or ht=0
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: Laws by Pries and Secomb (2005)
        "solver_option": 1,  # 1: Direct solver
                             # 2: PyAMG solver

        # Blood properties
        "ht_constant": 0.3,  # only required if RBC impact is considered
        "mu_plasma": 0.0012,

        # Zero Flow Vessel Threshold
        # True: the vessel with low flow are set to zero
        # The threshold is set as the max of mass flow balance
        # The function is reported in set_low_flow_threshold()
        "ZeroFlowThreshold": False ,
        "iterative_routine": 1,     # 1: Forward problem
                                    # 2: Iterative routine (ours)
                                    # 3: Iterative routine (Berg Thesis) [https://oatao.univ-toulouse.fr/25471/1/Berg_Maxime.pdf]
                                    # 4: Iterative routine (Rasmussen et al. 2018) [https://onlinelibrary.wiley.com/doi/10.1111/micc.12445]

        # Hexagonal network properties. Only required for "read_network_option" 1
        "nr_of_hexagon_x": 5,
        "nr_of_hexagon_y": 5,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 65],
        "hexa_boundary_values": [2, 1],
        "hexa_boundary_types": [1, 1],

        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "data/network/node_data.csv",
        "csv_path_edge_data": "data/network/edge_data.csv",
        "csv_path_boundary_data": "data/network/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "boundaryValue",

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": "C:/Users/manuf/Documents/2º DELFT/Intership/microBlooM/testcases/Graph_Manuel3.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": False,
        # Note: the extension of the output file is automatically added later in the function
        "write_path_igraph": "data/network/network_simulated",

        # Options for initializing the particles:
        "initial_number_particles": 8,
        "initial_vessels": [0,1,9,85,38,42, 70, 32], # same dimension as "initial_number_particles"
        "N_timesteps": 500,
        "interval_mode": 2,
        "use_tortuosity": 1,  # 0: Tortuosity off, 1: Tortuosity on
    }
)

# Create object to set up the simulation and initialise the simulation
setup_blood_flow = setup.SetupSimulation()
# Initialise the implementations based on the parameters specified
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver, imp_iterative, imp_balance = setup_blood_flow.setup_bloodflow_model(PARAMETERS)

# Build flownetwork object and pass the implementations of the different submodules, which were selected in
#  the parameter file
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, imp_iterative, imp_balance, PARAMETERS)

# Import or generate the network
print("Read network: ...")
flow_network.read_network()
print("Read network: DONE")

# Update the transmissibility
print("Update transmissibility: ...")
flow_network.update_transmissibility()
print("Update transmissibility: DONE")

# Update flow rate, pressure and RBC velocity
print("Update flow, pressure and velocity: ...")
flow_network.update_blood_flow()
print("Update flow, pressure and velocity: DONE")

# Check flow balance
print("Check flow balance: ...")
flow_network.check_flow_balance()
print("Check flow balance: DONE")

# Track the time for the particle-related steps
start_time_total = time.process_time()  # Start time for particle tracking process

# Initialization of particles
print("Initialization of particles into the network: ...")
start_initialization = time.process_time()
particle_tracker = Particle_tracker(PARAMETERS, flow_network)
initialization_time = time.process_time() - start_initialization
print(f"Initialization of particles into the network: DONE in {initialization_time:.4f} seconds")

# Simulation of particles
print("Simulation of particles into the network: ...")
start_simulation = time.process_time()
particles_evolution = particle_tracker.evolve_particles()
simulation_time = time.process_time() - start_simulation
print(f"Simulation of particles into the network: DONE in {simulation_time:.4f} seconds")

# Transformation to global coordinates
print("Transforming particles to global coordinates: ...")
start_transformation = time.process_time()
particles_evolution_global = particle_tracker.transform_to_global_coordinates()
transformation_time = time.process_time() - start_transformation
print(f"Transformation to global coordinates: DONE in {transformation_time:.4f} seconds")

# Define output directory for the VTK files
output_directory = "C:/Users/manuf/Documents/2º DELFT/Intership/microBlooM/data/network/output"

# Create VTK files per timestep
print("Creating VTK files for particles per timestep: ...")
start_vtk_creation = time.process_time()
particle_tracker.create_vtk_particles_per_timestep(particles_evolution_global, output_directory)
vtk_creation_time = time.process_time() - start_vtk_creation
print(f"VTK files created in directory: {output_directory} in {vtk_creation_time:.4f} seconds")

# Total time for particle processing
total_particle_process_time = time.process_time() - start_time_total
print(f"\nTotal time for particle processing: {total_particle_process_time:.4f} seconds")


