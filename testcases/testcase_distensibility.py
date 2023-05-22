"""
A python script to simulate stationary blood flow in microvascular networks with considering the vessel distensibility.
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
                                   # 3: import graph from igraph file (pickle file)
                                   # 4: todo import graph from edge_data and vertex_data pickle files
        "write_network_option": 1,  # 1: do not write anything
                                    # 2: write to igraph format # todo: handle overwriting data from import file
                                    # 3: write to vtp format
                                    # 4: write to two csv files
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
                                       # 3: todo: RBC tracking
                                       # 4-...: todo: steady state RBC laws
        "rbc_impact_option": 2,  # 1: No RBCs (hd=0)
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: Laws by Pries and Secomb (2005)
                                 # 4-...: todo: Other laws. in vivo?
        "solver_option": 1,  # 1: Direct solver
                             # 2: PyAMG solver
                             # 3-...: other solvers

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
        "csv_path_vertex_data": "data/network/b6_B_pre_061/node_data.csv",
        "csv_path_edge_data": "data/network/b6_B_pre_061/edge_data.csv",
        "csv_path_boundary_data": "data/network/b6_B_pre_061/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "p",

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": "not needed",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": False,  # todo: currently does not do anything
        "write_path_igraph": "data/network/b6_B_pre_061/results",  # only required for "write_network_option" 2, 3, 4

        ##########################
        # Vessel distensibility options
        ##########################

        # Set up distensibility model
        "distensibility_model": 2,   # 0: No update of diameters due to vessel distensibility
                                     # 1: Passive diameter changes, linearised. p_ext = p_base, d_ref = d_base
                                     # 2: Passive diameter changes, linearised. p_ext=0, d_ref computed.

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
flow_rate_base = np.abs(flow_network.flow_rate)
diameter_base = np.copy(flow_network.diameter)
pressure_base = .5 * np.sum(flow_network.pressure[flow_network.edge_list], axis=1) / 133.3

# Post stroke
distensibility.initialise_distensibility()
stroke_edges = np.array([0, 1])

distensibility.diameter_ref = np.delete(distensibility.diameter_ref, stroke_edges)
distensibility.e_modulus = np.delete(distensibility.e_modulus, stroke_edges)
distensibility.wall_thickness = np.delete(distensibility.wall_thickness, stroke_edges)
distensibility.eid_vessel_distensibility = np.delete(distensibility.eid_vessel_distensibility, stroke_edges)
distensibility.nr_of_edge_distensibilities = np.size(distensibility.eid_vessel_distensibility)

flow_network.diameter[stroke_edges] = .5e-6

for i in range(100):
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    distensibility.update_vessel_diameters()

flow_rate_stroke = np.abs(flow_network.flow_rate)
diameter_stroke = np.copy(flow_network.diameter)
pressure_stroke = .5 * np.sum(flow_network.pressure[flow_network.edge_list], axis=1) / 133.3

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Plot network

flow_rates_rel_change = (flow_rate_stroke - flow_rate_base) / flow_rate_base

xy = flow_network.xyz[:, :2]
edgelist = flow_network.edge_list

segments = xy[edgelist]

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(xy[:, 0].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 0].max() + PARAMETERS['hexa_edge_length'])
ax.set_ylim(xy[:, 1].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 1].max() + PARAMETERS['hexa_edge_length'])
line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"),
                               norm=plt.Normalize(vmin=-1, vmax=0))
line_segments.set_array(flow_rates_rel_change)
line_segments.set_linewidth(diameter_stroke * 1e5*2)
ax.add_collection(line_segments)

ax.set_title('Flow rate changes after stroke')

ax.plot(xy[flow_network.boundary_vs, 0], xy[flow_network.boundary_vs, 1], 'o', color='r', markersize=10)

cbar1 = plt.colorbar(line_segments)

fig.savefig("output/flowrates_change.png", dpi=200)


diameter_rel_change = (diameter_stroke-diameter_base)/diameter_base
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(xy[:, 0].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 0].max() + PARAMETERS['hexa_edge_length'])
ax.set_ylim(xy[:, 1].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 1].max() + PARAMETERS['hexa_edge_length'])
line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"),
                               norm=plt.Normalize(vmin=-.2, vmax=0))
line_segments.set_array(diameter_rel_change)
line_segments.set_linewidth(diameter_stroke * 1e5*2)
ax.add_collection(line_segments)

ax.set_title('Diameter changes after stroke')

ax.plot(xy[flow_network.boundary_vs, 0], xy[flow_network.boundary_vs, 1], 'o', color='r', markersize=10)

cbar1 = plt.colorbar(line_segments)

fig.savefig("output/diameter_change.png", dpi=200)


fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(xy[:, 0].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 0].max() + PARAMETERS['hexa_edge_length'])
ax.set_ylim(xy[:, 1].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 1].max() + PARAMETERS['hexa_edge_length'])
line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"),
                               norm=plt.Normalize(vmin=10, vmax=100))
line_segments.set_array(pressure_stroke)
line_segments.set_linewidth(diameter_stroke * 1e5*2)
ax.add_collection(line_segments)

ax.set_title('Pressure after stroke')

ax.plot(xy[flow_network.boundary_vs, 0], xy[flow_network.boundary_vs, 1], 'o', color='r', markersize=10)

cbar1 = plt.colorbar(line_segments)

fig.savefig("output/pressure_stroke.png", dpi=200)


fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(xy[:, 0].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 0].max() + PARAMETERS['hexa_edge_length'])
ax.set_ylim(xy[:, 1].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 1].max() + PARAMETERS['hexa_edge_length'])
line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"),
                               norm=plt.Normalize(vmin=10, vmax=100))
line_segments.set_array(pressure_base)
line_segments.set_linewidth(diameter_base * 1e5*2)
ax.add_collection(line_segments)

ax.set_title('Pressure baseline')

ax.plot(xy[flow_network.boundary_vs, 0], xy[flow_network.boundary_vs, 1], 'o', color='r', markersize=10)

cbar1 = plt.colorbar(line_segments)

fig.savefig("output/pressure_base.png", dpi=200)


