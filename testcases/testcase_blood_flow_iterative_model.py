"""
A python script to simulate stationary blood flow in microvascular networks. Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Save the results in a file
"""
import numpy as np

from source.flow_network import FlowNetwork
from types import MappingProxyType
import source.setup.setup as setup

# set random seed
#np.random.seed(42)

# MappingProxyType is basically a const dict.
# todo: read parameters from file; need a good way to import from human readable file (problem: json does not support
#  comments, which we would like to have in the text file; need better solution...)
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 2,  # 1: generate hexagonal graph
        # 2: import graph from csv files
        # 3: import graph from igraph file (pickle file)
        # 4: generate hexagonal graph composed of a single hexagon
        # 5: generate hexagonal graph composed of a single hexagon for trifurcation
        # 6: todo import graph from edge_data and vertex_data pickle files
        "write_network_option": 1,  # 1: do not write anything
        # 2: write to igraph format # todo: handle overwriting data from import file
        # 3: write to vpt format
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
        # 2: Constant hematocrit
        # 3: todo: RBC tracking
        # 4-...: todo: steady state RBC laws
        "rbc_impact_option": 4,  # 1: No RBCs (hd=0)
        # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
        # 3: Laws by Pries and Secomb (2005)
        # 4: Laws by Preis (1996)
        "solver_option": 1,  # 1: Direct solver
        # 2: PyAMG solvervisual
        # 3-...: other solvers
        "iterative_routine": 2,  # 1: Single iteration
        # 2: Iterative routine

        # Blood properties
        "ht_constant": 4E-05,  # only required if RBC impact is considered
        "mu_plasma": 0.0052,
        "boundary_hematocrit": 0.1,  # np.random.uniform(low=0.09, high=0.11, size=2000),  # np.full(2000, 0.1)
        # #: np.array([0.1] * 2000),  #
        "network_name": "MVN1_01_prova_incremento",

        # if True set the blood vessel with unrealistic blood flow to zero
        "low_flow_vessel": True,

        # Machine error for float
        "machine_error": 1E-15,

        # Alpha for relaxation factor of SOR
        "alpha": 1,
        "epsilon": 5E-25,
        "epsilon_second_method": 1E-10,

        # zeroFlowThreshold default is False, true inc ase of iterative routine
        "zeroFlowThreshold": False,
        # Hexagonal network properties. Only required for "read_network_option" 1
        "nr_of_hexagon_x": 5,
        "nr_of_hexagon_y": 5,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 5.e-6,
        "hexa_boundary_vertices": [0, 27],
        "hexa_boundary_values": [5, 2],
        "hexa_boundary_types": [1, 1],

        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "MVN2/node_data.csv",
        "csv_path_edge_data": "MVN2/edge_data.csv",
        "csv_path_boundary_data": "MVN2/node_boundary_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeID", "csv_boundary_type": "boundaryType", "csv_boundary_value": "p",

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": "B6_B_01/b6_B_pre_stroke.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: fl<ow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": True,  # todo: currently does not do anything
        "write_path_igraph": "data/out/9_11/paraview/MVN1_01_prova_incremento.vtp",
        # only required for "write_network_option" 2
        "save": True,
        "path_for_graph": "data/out/9_11/MVN1_01_prova_incremento/plot",

        # Write option in case of print in output file (.txt)
        "path_output_file": "data/out/9_11/MVN1_01_prova_incremento/log_file",

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

# Write the results to file
flow_network.write_network()
