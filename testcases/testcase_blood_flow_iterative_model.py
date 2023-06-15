"""
A python script to simulate stationary blood flow in microvascular networks. Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Save the results in a file
"""
import copy
import sys
import numpy as np
import pandas as pd

import util_methods
from source.flow_network import FlowNetwork
from source.bloodflowmodel.flow_balance import FlowBalance
from types import MappingProxyType
import source.setup.setup as setup
from util_methods.display_graph import display_graph_util
from util_methods.util_iterative import util_iterative_method

# MappingProxyType is basically a const dict.
# todo: read parameters from file; need a good way to import from human readable file (problem: json does not support
#  comments, which we would like to have in the text file; need better solution...)
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 1,  # 1: generate hexagonal graph
        # 2: import graph from csv files
        # 3: import graph from igraph file (pickle file)
        # 4: generete hexagonal graph composed of a single hexagon
        # 5: generete hexagonal graph composed of a single hexagon for trifurcation
        # 6: todo import graph from edge_data and vertex_data pickle files
        "write_network_option": 2,  # 1: do not write anything
        # 2: write to igraph format # todo: handle overwriting data from import file
        # 3-...: todo other file formats.
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
        # 2: Constant haematocrit
        # 3: todo: RBC tracking
        # 4-...: todo: steady state RBC laws
        "rbc_impact_option": 4,  # 1: No RBCs (hd=0)
        # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
        # 3: Laws by Pries and Secomb (2005)
        # 4: Laws by Preis (1996)
        "solver_option": 1,  # 1: Direct solver
        # 2: PyAMG solver
        # 3-...: other solvers
        "convergence_case": 2,
        # 1: Difference between two iteration of the qRBCs
        # 2: Normalized mean absolute error

        # Blood properties
        "ht_constant": 5E-01,  # only required if RBC impact is considered
        "mu_plasma": 0.0012,
        "boundary_hematocrit": 3E-01,
        # alpha
        "alpha": 0.2,
        "epsilon": 5E-25,
        "epsilon_second_method": 1E-6,

        # Hexagonal network properties. Only required for "read_network_option" 1
        "nr_of_hexagon_x": 3,
        "nr_of_hexagon_y": 3,
        "hexa_edge_length": 62.E-6,
        "hexa_diameter": 5e-6,
        "hexa_boundary_vertices": [0, 6, 27],
        "hexa_boundary_values": [4, 3, 1],
        "hexa_boundary_types": [1, 1, 1],

        # implementation for multiple inflows
        "boundary_vertices_in_out": [0, 0, 1],  # 0 for in 1 for out
        "boundary_vertices_in_hem": [5E-01, 3E-01],
        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "data/network/b6_B_pre_061/node_data.csv",
        "csv_path_edge_data": "data/network/b6_B_pre_061/edge_data.csv",
        "csv_path_boundary_data": "data/network/b6_B_pre_061/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "boundaryValue",

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": "/Users/cucciolo/Desktop/Bern/GraphHoney.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": True,  # todo: currently does not do anything
        "write_path_igraph": "/Users/cucciolo/Desktop/microBlooM/data/out/hematocrit.pkl",
        # only required for "write_network_option" 2
        "save": False,
        "path_for_graph": "/Users/cucciolo/Desktop/microBlooM/data/out",
    }
)

# Create object to set up the simulation and initialise the simulation
setup_blood_flow = setup.SetupSimulation()
# Initialise the implementations based on the parameters specified
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, imp_solver = setup_blood_flow.setup_bloodflow_model(
    PARAMETERS)

# Build flownetwork object and pass the implementations of the different submodules, which were selected in
#  the parameter file
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)
flow_balance = FlowBalance(flow_network)

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

print("Iterative Approach: ...")
util_iterative_method(PARAMETERS, flow_network)
print("Iterative Approach: DONE")

print("Check flow balance: ...")
flow_balance.check_flow_balance()
print("Check flow balance: DONE")

# Write the results to file
flow_network.write_network()
