
import numpy as np

from source.flow_network import FlowNetwork
from types import MappingProxyType
from source.particle_tracking.particle_tracking import Particle_tracker
import source.setup.setup as setup

# MappingProxyType is basically a const dict.
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 1,  # 1: generate hexagonal graph
                                   # 3: import graph from igraph format (pickle file)
        "write_network_option": 1,  # 1: do not write anything
                                    # 2: write to igraph format (.pkl)
                                    # 3: write to vtp format (.vtp)
                                    # 4: write to two csv files (.csv)
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
                                       # 3: Hematocrit computed based on number of particles in each vessel
        "rbc_impact_option": 3,  # 1: No RBCs (hd=0) - makes only sense if tube_haematocrit_option:1 or ht=0
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: Laws by Pries and Secomb (2005)
        "solver_option": 1,  # 1: Direct solver
                             # 2: PyAMG solver
                             # 3: Pardiso solver
        "iterative_routine": 1,  # 1: Forward problem
                                 # 2: Iterative routine (ours)
                                 # 3: Iterative routine (Berg Thesis) [https://oatao.univ-toulouse.fr/25471/1/Berg_Maxime.pdf]
                                 # 4: Iterative routine (Rasmussen et al. 2018) [https://onlinelibrary.wiley.com/doi/10.1111/micc.12445]

        # Elastic vessel - vascular properties (tube law) - Only required for distensibility and autoregulation models
        "pressure_external": 0.,  # Constant external pressure
        "read_vascular_properties_option": 1,  # 1: Do not read anything
        "tube_law_ref_state_option": 1,  # 1: No compute of reference diameters (d_ref)
        "csv_path_vascular_properties": "not_needed",  # Young's Modulus and Wall Thickness for all vessels

        # Blood properties
        "ht_constant": 0.3,  # only required if RBC impact is considered
        "mu_plasma": 0.0012,

        # Zero Flow Vessel Threshold
        # True: the vessel with low flow are set to zero
        # The threshold is set as the max of mass flow balance
        # The function is reported in set_low_flow_threshold()
        "ZeroFlowThreshold": False,

        # Hexagonal network properties. Only required for "read_network_option" 1
        "nr_of_hexagon_x": 3,
        "nr_of_hexagon_y": 3,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 27],
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
        "pkl_path_igraph": "data/network/network_graph.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": False,
        # Note: the extension of the output file is automatically added later in the function
        "write_path_igraph": "data/network/network_simulated",

        # OPTIONS for Particle tracking:
        "particles_type": 0, # 0 = passive particles (don't affect the flowfiled), 1 = RBCs
                             # IMPORTANT!!!: for RBC usage set "tube_haematocrit_option" = 3
        "N_timesteps": 800, # Number of timesteps the simulations has to be run (iterations saved as output)
        "rbc_volume": 4.9e-17, # Volume of the particles you are introducing. In the case of passive particle
                               # is also needed for initialization. Common value for mice: 4.9e-17 m^3.
        "ht_initial": 0.01, # Initial hematocrit in the network. Controls amount of particles introduced 
                           # in the network before running the simulation. Also used in passive particle case
                           # for initialization. This value has to be grater than zero !!! At least one particle has to be initialized
                           # in the netowrk.
        "ht_boundary_condition":0.1, # Controls amount of particles introduced in the network at every timestep. 
                                     # Simulates an external microvasculature connected to the inflow nodes with
                                     # average constant hematocrit equal to ht_boundary_condition.
        "times_basic_delta_t": 6,   # The basic timestep is computed as the minimum vessel length divided by
                                    # the maximum rbc_velocity. The timestep used is computed as:
                                    #   delta_t = times_basic_delta_t * basic_timestep (adaptative timestep)

        "preinitialize_with_iterations": 1, # Option for running some iterations based on characteristic time
                                            # before computing the actual simulation. 0 = off, 1 = on
        "times_Tc_preinitialization": 0.5, # Only affects if "preinitialize_with_iterations" = 1.
                                         # Controls amount of timesteps for the preinitialization defined as
                                         # N_timesteps_preinit = K * Characteristic_time. K is the user choice here.
                                         # Characteristic_time = (Total_blood_volume) [m^3] / (Outflow_rate) [m^3 / s]
        "timestep_type_after_preinitialization": 0, # 0 = fixed timestep to be specified in "delta_t_after_preinitialization"
                                                    # 1 = adaptative timestep 
        "delta_t_after_preinitialization": 0.0005, # Only affects if "preinitialize_with_iterations" = 1
                                                    # and "timestep_type_after_preinitialization"=  0
                                                    # Timestep in seconds after the pre-initialize iterations
       
        "use_tortuosity": 1,  # 0: Tortuosity off, 1: Tortuosity on. ONLY avaialable if: "read_network_option" = 3
        "parallel": False,  # Set to True for parallel execution, False for sequential
                          # NOTE: For running the parallel version the user should:
                          #          1- Have an MPI implementation installed on the system.
                          #          2- Have 'mpi4py' Python package installed in the used Python interpreter.
                          #          2- Execute in the terminal: 'mpiexec -np x python main.py'
                          #             Where x is the number of processes selected.   
        
        # Output control (0 = off, 1 = on)
        "output_directory": "output/particle_tracking_output", # Folder for saving output files
        "output_particles_evolution": 1,   # CSV with (vessel, alpha) for every particle for every timestep
        "output_vessel_evolution": 1,      # CSV with vessel for every particle for every timestep
        "output_velocity_components": 1,   # compute velocity components per particle and timestep + save in csv files. Only available if "use_tortuosity"= 1.
        "compute_global_coords": 1,        # compute global coordiantes per particle and timestep   
        "save_global_coords": 1,           # ONLY if "compute_global_coords" = 1. save global coordinates per particle and timestep in csv files
        "output_vtp_files": 0,             # ONLY if "compute_global_coords" = 1. Creates 1 .vtp file per timestep for Paraview visualization

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
setup_simulation = setup.SetupSimulation()
# Initialise the implementations based on the parameters specified
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver, imp_iterative, imp_balance, imp_read_vascular_properties, imp_tube_law_ref_state = setup_simulation.setup_bloodflow_model(PARAMETERS)

# Build flownetwork object and pass the implementations of the different submodules, which were selected in
#  the parameter file
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, imp_iterative, imp_balance, imp_read_vascular_properties,
                           imp_tube_law_ref_state, PARAMETERS)

# Import or generate the network
if rank == 0:
    print("Read network: ...")
flow_network.read_network()
if PARAMETERS["particles_type"] == 1:
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