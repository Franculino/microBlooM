"""
A python script to simulate stationary blood flow in microvascular networks considering the vessel distensibility.
Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Update the vessel diameters based on the current pressure distribution
5. Save the results in a file
"""
import sys
import numpy as np

from source.flow_network import FlowNetwork
from source.distensibility import Distensibility
from types import MappingProxyType
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

        # Hexagonal network properties. Only required for "read_network_option" 1
        "nr_of_hexagon_x": 9,
        "nr_of_hexagon_y": 9,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4e-6,
        "hexa_boundary_vertices": [0, 189],
        "hexa_boundary_values": [13330, 1333],
        "hexa_boundary_types": [1, 1],

        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "data/network/node_data.csv",
        "csv_path_edge_data": "data/network/edge_data.csv",
        "csv_path_boundary_data": "data/network/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "p",

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": "data/network/network_graph.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": False,
        # Note: the extension of the output file is automatically added later in the function
        "write_path_igraph": "data/network/network_simulated",

        ################################
        # Vessel distensibility options
        ################################

        # Set up distensibility model
        "distensibility_model": 3,   # 1: No update of diameters due to vessel distensibility
                                     # 2: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = p_base, d_ref = d_base
                                     # 3: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const, d_ref computed.

        # Distensibility edge properties
        "csv_path_distensibility": "data/distensibility/distensibility_parameters.csv",
        "pressure_external": 0.  # Constant external pressure as reference pressure (only for distensibility_model 2)
    }
)

# Create object to set up the simulation and initialise the simulation
setup_blood_flow = setup.SetupSimulation()
# Initialise the implementations based on the parameters specified
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver = setup_blood_flow.setup_bloodflow_model(PARAMETERS)

imp_distensibility_law, imp_read_distensibility_parameters = setup_blood_flow.setup_distensibility_model(PARAMETERS)

# Build flownetwork object and pass the implementations of the different submodules, which were selected in
#  the parameter file
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)

distensibility = Distensibility(flow_network, imp_distensibility_law, imp_read_distensibility_parameters)

# Import or generate the network
print("Read network: ...")
flow_network.read_network()
print("Read network: DONE")

# Baseline
flow_network.update_transmissibility()
flow_network.update_blood_flow()

# Post stroke
distensibility.initialise_distensibility()
stroke_edges = np.array([0, 1])  # Example: Occlude 2 edges at inflow
flow_network.diameter[stroke_edges] = .5e-6

distensibility.diameter_ref = np.delete(distensibility.diameter_ref, stroke_edges)
distensibility.e_modulus = np.delete(distensibility.e_modulus, stroke_edges)
distensibility.wall_thickness = np.delete(distensibility.wall_thickness, stroke_edges)
distensibility.eid_vessel_distensibility = np.delete(distensibility.eid_vessel_distensibility, stroke_edges)
distensibility.nr_of_edge_distensibilities = np.size(distensibility.eid_vessel_distensibility)

# Update diameters and iterate
for i in range(100):
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    distensibility.update_vessel_diameters()

flow_network.write_network()
