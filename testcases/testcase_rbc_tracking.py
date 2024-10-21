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
from source.particle_tracking.rbc_tracking import Particle_tracker
import source.setup.setup as setup

# MappingProxyType is basically a const dict.
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 3,  # 1: generate hexagonal graph
                                   # 2: import graph from csv files
                                   # 3: import graph from igraph format (pickle file)
        "write_network_option": 3,  # 1: do not write anything
                                    # 2: write to igraph format (.pkl)
                                    # 3: write to vtp format (.vtp)
                                    # 4: write to two csv files (.csv)
        "tube_haematocrit_option": 3,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
                                       # 3: Hematocrit computed based on number of particles in each vessel
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
        "nr_of_hexagon_x": 11,
        "nr_of_hexagon_y": 11,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0,275],
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
        "pkl_path_igraph": "C:/Users/manuf/Documents/2º DELFT/Intership/microBlooM/testcases/piece_corrected_SI.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": False,
        # Note: the extension of the output file is automatically added later in the function
        "write_path_igraph": "data/network/network_simulated",

        # Options for initializing the particles:
        "initial_particles_mode": 0, # 0: use initial_number_particles and initial_vessels to set the initial positions
                                     # 1: use initial_hematocrit to stablish a constant initial hematocrit in all the network
                                     #    that will determine the intial number of particles in each vessel
        "initial_number_particles": 8,
        "initial_vessels": [0,1,9,85,38,42, 70, 32], # same dimension as "initial_number_particles"
        "ht_initial": 0.4, 
        "rbc_volume": 4.9e-17,
        "N_timesteps": 300,
        "times_basic_delta_t":10,   # The basic timestep is computed as the minimum vessel length divided by
                                    # the maximum rbc_velocity. The timestep used is computed as:
                                    #   delta_t = times_basic_delta_t * basic_timestep

        "interval_mode": 1, # 0: the same inflowing frequency in every inflowing vertex
                            # 1: inflowing frequency based on flow_rate of vessels connected to each vertex
        "particles_frequency": 1, # Only required if "interval_mode" 0. Every "particles_freq" 
                                  # timesteps a particle will enter through each inflow. Minimum possible value: 1. 
        "use_tortuosity": 1,  # 0: Tortuosity off, 1: Tortuosity on
        "parallel": False  # Set to True for parallel execution, False for sequential
                          # NOTE: For running the parallel version the user should:
                          #          1- Have an MPI implementation installed on the system.
                          #          2- Have 'mpi4py' Python package installed in the used Python interpreter.
                          #          2- Execute in the terminal: 'mpiexec -np x python main.py'
                          #             Where -np is the number of processe selected. 

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

if PARAMETERS['parallel']:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    rank = 0
    size = 1


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
if rank == 0:
    print("Read network: ...")

flow_network.read_network()
flow_network.num_particles_in_vessel = np.zeros(len(flow_network.edge_list), dtype=int)
flow_network.volume = np.ones(len(flow_network.edge_list), dtype=int)
if rank == 0:
    print("Read network: DONE")

# Update the transmissibility
if rank == 0:    
    print("Update transmissibility: ...")

flow_network.update_transmissibility()

if rank == 0:    
    print("Update transmissibility: DONE")

# Update flow rate, pressure and RBC velocity
if rank == 0:
    print("Update flow, pressure and velocity: ...")

flow_network.update_blood_flow()

if rank == 0:
    print("Update flow, pressure and velocity: DONE")

# Check flow balance
if rank == 0: 
    print("Check flow balance: ...")

flow_network.check_flow_balance()
flow_network.write_network()

if rank == 0:
    print("Check flow balance: DONE")

# Track the time for the particle-related steps
start_time_total = time.process_time()  # Start time for particle tracking process

# Initialization of particles
if rank == 0:
    print("Initialization of particles into the network: ...")

start_initialization = time.process_time()
particle_tracker = Particle_tracker(PARAMETERS, flow_network)
initialization_time = time.process_time() - start_initialization

if rank == 0:
    print(f"Initialization of particles into the network: DONE in {initialization_time:.4f} seconds")

# Simulation of particles
if rank == 0:
    print("Simulation of particles into the network: ...")

start_simulation = time.process_time()
particles_evolution = particle_tracker.evolve_particles()
simulation_time = time.process_time() - start_simulation

if rank == 0:
    print(f"Simulation of particles into the network: DONE in {simulation_time:.4f} seconds")

# Transformation to global coordinates
if rank == 0:
    print("Transforming particles to global coordinates: ...")
start_transformation = time.process_time()

if PARAMETERS['parallel']:
    comm.Barrier()
    particles_evolution_global = particle_tracker.transform_to_global_coordinates()
    comm.Barrier()
else:
    particles_evolution_global = particle_tracker.transform_to_global_coordinates()

if rank == 0:
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