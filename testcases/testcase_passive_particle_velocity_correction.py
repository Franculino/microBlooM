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
from source.particle_tracking.passive_particle_velocity_correction import Particle_tracker
import source.setup.setup as setup

# MappingProxyType is basically a const dict.
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 1,  # 1: generate hexagonal graph
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
        "nr_of_hexagon_x": 11,
        "nr_of_hexagon_y": 11,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 14, 275],
        "hexa_boundary_values": [2, 2, 1],
        "hexa_boundary_types": [1,1, 1],

        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "data/network/node_data.csv",
        "csv_path_edge_data": "data/network/edge_data.csv",
        "csv_path_boundary_data": "data/network/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "boundaryValue",

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": "./testcases/MVN2_corrected_SI.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": False,
        # Note: the extension of the output file is automatically added later in the function
        "write_path_igraph": "data/network/network_simulated",

        # Options for Particle tracking:
        
        "N_timesteps": 800,
        "rbc_volume": 4.9e-17,
        "ht_initial": 0.1,
        "ht_boundary_condition":0.5,
        "times_basic_delta_t": 1,   # The basic timestep is computed as the minimum vessel length divided by
                                    # the maximum rbc_velocity. The timestep used is computed as:
                                    #   delta_t = times_basic_delta_t * basic_timestep
        "preinitialize_with_iterations": 0, 
        "times_Tc_forsteadystate": 0.02,
        "delta_t_after_steady_state": 0.0002,
        "use_tortuosity": 0,  # 0: Tortuosity off, 1: Tortuosity on
        "parallel": False,  # Set to True for parallel execution, False for sequential
                          # NOTE: For running the parallel version the user should:
                          #          1- Have an MPI implementation installed on the system.
                          #          2- Have 'mpi4py' Python package installed in the used Python interpreter.
                          #          2- Execute in the terminal: 'mpiexec -np x python main.py'
                          #             Where -np is the number of processe selected.   
        
        # Output control (0 = off, 1 = on)
        "output_directory": "data/network/output",
        "output_particles_evolution": 1,   # CSV with (vessel, alpha) for every particle for every timestep
        "output_vessel_evolution": 1,      # CSV with vessel for every particle for every timestep
        "output_velocity_components": 0,   # compute velocity components per particle and timestep + save in csv files
        "output_nkind_matrix": 0,          # compute matrix indicating type of vessel per particle and timestep + save in csv file
        "compute_global_coords": 1,        # compute global coordiantes per particle and timestep   
        "save_global_coords": 1,           # ONLY if "compute_global_coords" = 1. save global coordinates per particle and timestep in csv files
        "output_vtp_files": 1,             # ONLY if "compute_global_coords" = 1. Creates 1 .vtp file per timestep for Paraview visualization

    }
)
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
if rank == 0:
    print("Check flow balance: DONE")

# Initialization of particles
if rank == 0:
    print("Initialization of particles into the network: ...")
particle_tracker = Particle_tracker(PARAMETERS, flow_network)
if rank == 0:
    print(f"Initialization of particles into the network: DONE ")

# Simulation of particles
if rank == 0:
    if PARAMETERS["preinitialize_with_iterations"] == 1:
        print("Simulation of particles into the network: ...")
        particles_evolution_steadystate = particle_tracker.evolve_particles()
        particle_tracker.save_steady_state()
        particle_tracker.initialize_from_steady_state()
        particles_evolution = particle_tracker.evolve_particles()
        print(f"Simulation of particles into the network: DONE ")
    if PARAMETERS["preinitialize_with_iterations"] == 0:
        print("Simulation of particles into the network: ...")
        particles_evolution = particle_tracker.evolve_particles()
        print(f"Simulation of particles into the network: DONE ")
if rank == 0:
    
    print("Write network: ...")
    flow_network.write_network()
    print("Write network: DONE")

if rank == 0:
    print(f"Simulation of particles into the network: DONE ")

# Transformation to global coordinates
if PARAMETERS["compute_global_coords"] == 1:
    if rank == 0:
        print("Transforming particles to global coordinates: ...")

    if PARAMETERS['parallel']:
        comm.Barrier()
        particles_evolution_global = particle_tracker.transform_to_global_coordinates()
        comm.Barrier()
    else:
        particles_evolution_global = particle_tracker.transform_to_global_coordinates()
if rank == 0:
    print(f"Transformation to global coordinates: DONE ")

# OUTPUT generation
if rank == 0:
    # Define output directory for the VTK files
    output_directory = "C:/Users/manuf/OneDrive - Delft University of Technology/Documenten/2_DELFT/Internship/microBlooM/data/network/output"
    print("Generating requested outputs...")
    particle_tracker.generate_outputs()
    print("All requested outputs have been generated.")

