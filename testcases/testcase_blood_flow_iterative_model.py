"""
A python script to simulate stationary blood flow in microvascular networks. Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Save the results in a file
"""

from source.flow_network import FlowNetwork
from types import MappingProxyType
import source.setup.setup as setup


# MappingProxyType is basically a const dict.
# todo: read parameters from file; need a good way to import from human readable file (problem: json does not support
#  comments, which we would like to have in the text file; need better solution...)
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for the blood flow model
        "read_network_option": 1,   # 1: generate hexagonal graph
                                    # 2: import graph from csv files
                                    # 3: import graph from igraph file (pickle file)
                                    # ...: todo import graph from edge_data and vertex_data pickle files
        "write_network_option": 1,  # 1: do not write anything
                                    # 2: write to igraph format # todo: handle overwriting data from import file
                                    # 3: write to vpt format
                                    # 4: write to two csv files (.csv)
        "tube_haematocrit_option": 2,   # 1: No RBCs (ht=0)
                                        # 2: Constant hematocrit
                                        # 3: todo: RBC tracking
                                        # ...: todo: steady state RBC laws
        "rbc_impact_option": 4,     # 1: No RBCs (hd=0)
                                    # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                    # 3: Laws by Pries and Secomb (2005)
                                    # 4: Laws for phase separation by Preis (1996)
        "solver_option": 1,     # 1: Direct solver
                                # 2: PyAMG solver visual
                                # ...: other solvers
        "iterative_routine": 2,     # 1: Forward problem
                                    # 2: Iterative routine (ours)
                                    # 3: Iterative routine (Berg Thesis) [https://oatao.univ-toulouse.fr/25471/1/Berg_Maxime.pdf]
                                    # 4: Iterative routine (Rasmussen et al. 2018) [https://onlinelibrary.wiley.com/doi/10.1111/micc.12445]

        # Blood properties
        "ht_constant": 4E-05,  # only required if RBC impact is considered
        "mu_plasma": 0.0052,
        "boundary_hematocrit": 0.4,
        "network_name": "Network_001", 

        # Zero Flow Vessel Threshold
        # True: the vessel with low flow are set to zero
        # The threshold is set as the max of mass-flow balance
        # The function is reported in set_low_flow_threshold()        
        "ZeroFlowThreshold": True,

        # Hexagonal network properties. Only required for "read_network_option" 1
        # For options 4-6 the hexagonal nr_of_hexagon_x and nr_of_hexagon_y are not needed
        "nr_of_hexagon_x": 3,
        "nr_of_hexagon_y": 5,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 27],
        "hexa_boundary_values": [2, 1],
        "hexa_boundary_types": [1, 1],

        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "data/network/node_data.csv",
        "csv_path_edge_data": "data/network/edge_data.csv"",
        "csv_path_boundary_data": "data/network/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "boundaryValue", 

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": "B6_B_01/b6_B_pre_stroke.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": False,  # todo: currently does not do anything
        "write_path_results": "data/network/results",
        # only required for "write_network_option" 2
        "save": True,
        "path_for_graph": "data/output/plot/",

        # Write option for iterative routine - Output files for convergence check
        "path_output_file": "data/output/log_file/",  # path to output the csv/pckl files
        # save central data in a pckl 
        "pckl_save": False

    }
)

# Create an object to set up the simulation and initialise the simulation
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
# Check flow balance
print("Iterative Routine: ...")
flow_network.update_blood_flow()
print("Iterative Routine: DONE")

# Write the results to file
flow_network.write_network()
