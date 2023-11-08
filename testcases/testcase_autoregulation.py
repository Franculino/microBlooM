"""
A python script to simulate stationary blood flow in microvascular networks with considering the vessel distensibility
in order to reconstruct the passive autoregulation curve.
At healthy conditions, the inlet boundary pressure is changing only from 50 to 180 mmHg, with the baseline at 100mmHg.
The reference state for the distensibility law is computed based on the baseline condition (100mmHg).
Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Update the vessel diameters based on the current pressure distribution
5. Save the results in a file
"""
import sys
import numpy as np
import pandas as pd

from source.flow_network import FlowNetwork
from source.bloodflowmodel.flow_balance import FlowBalance
from source.distensibility import Distensibility
from types import MappingProxyType
import source.setup.setup as setup


# MappingProxyType is basically a const dict.
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 3,  # 1: generate hexagonal graph
                                   # 2: import graph from csv files
                                   # 3: import graph from igraph file (pickle file)
                                   # 4: todo import graph from edge_data and vertex_data pickle files
        "write_network_option": 4,  # 1: do not write anything
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
        "stroke_edges": [0, 1],  # Example: Occlude 2 edges at inflow
        "diameter_stroke_edges": .5e-6,

        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "not needed",
        "csv_path_edge_data": "not needed",
        "csv_path_boundary_data": "not needed",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "p",

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": "testcase_autoregulation/passive_autoregulation_curve/data/B6_B_init_061/B6_B_init_061.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_path_igraph": "testcase_autoregulation/passive_autoregulation_curve/output/B6_C_init_001/pin120/E_modulus_constant/B6_C_init_001_results",  # only required for "write_network_option" 2, 3, 4

        ##########################
        # Vessel distensibility options
        ##########################

        # Set up distensibility model
        "read_dist_parameters_option": 2,       # 1: Do not read anything
                                                # 2: Read from csv file

        "distensibility_ref_state_option": 3,   # 1: No update of diameters due to vessel distensibility
                                                # 2: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = p_base,
                                                    # d_ref = d_base
                                                # 3: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const,
                                                    # d_ref computed based on Sherwin et al. (2003)
                                                # 4: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const,
                                                    # d_ref computed based on Urquiza et al. (2006)
                                                # 5: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const,
                                                    # d_ref computed based on Rammos et al. (1998)

        "distensibility_relation_option": 2,    # 1: No update of diameters due to vessel distensibility
                                                # 2: Relation based on Sherwin et al. (2003) - non linear p-A relation
                                                # 3: Relation based on Urquiza et al. (2006) - non linear p-A relation
                                                # 4: Relation based on Rammos et al. (1998) - linear p-A relation

        # Distensibility edge properties
        "csv_path_distensibility": "testcase_autoregulation/passive_autoregulation_curve/data/B6_C_init_001/B6_C_init_001_conE_dist_properties.csv",
        "pressure_external": 0.,  # Constant external pressure as reference pressure (only for distensibility_model 2)

    }
)

print("pkl_path_igraph", PARAMETERS["pkl_path_igraph"])
# Create object to set up the simulation and initialise the simulation
setup_blood_flow = setup.SetupSimulation()
# Initialise the implementations based on the parameters specified
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver = setup_blood_flow.setup_bloodflow_model(PARAMETERS)

imp_read_distensibility_parameters, imp_distensibility_ref_state, imp_distensibility_relation = \
    setup_blood_flow.setup_distensibility_model(PARAMETERS)

# Build flownetwork object and pass the implementations of the different submodules, which were selected in
#  the parameter file
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)

flow_balance = FlowBalance(flow_network)

distensibility = Distensibility(flow_network, imp_distensibility_ref_state, imp_read_distensibility_parameters,
                                imp_distensibility_relation)

# Import or generate the network - Import data for the pre-stroke state
print("Read network: ...")
flow_network.read_network()
print("Read network: DONE")

# Baseline
# Diameters at baseline Boundary conditions: inlet-100mmHg and outlet-10mmHg.
# They are needed to compute the reference pressure and diameters - only for distensibility_ref_state: 3, 4, 5
print("Solve baseline flow (for baseline boundary conditions): ...")
flow_network.update_transmissibility()
flow_network.update_blood_flow()
print("Solve baseline flow (for baseline boundary conditions): DONE")

print("Check flow balance: ...")
flow_balance.check_flow_balance()
print("Check flow balance: DONE")

# Initialise distensibility model based on baseline diameters and pressures
print("Initialise distensibility model based on baseline results: ...")
distensibility.initialise_distensibility()
print("Initialise distensibility model based on baseline results: DONE")

# Change the intel pressure boundary condition
print("Change the intel pressure boundary condition: ...")
bc_current = pd.read_csv("testcase_autoregulation/passive_autoregulation_curve/data/B6_C_init_001/B6_C_init_001_pin120.csv")["boundaryValue"].to_numpy()
flow_network.boundary_val = bc_current
print("Change the intel pressure boundary condition: DONE")

# Update diameters and iterate (has to be improved)
print("Update the diameters based on Distensibility Law: ...")
tol = 1.e-10
diameters_current = flow_network.diameter  # Previous diameters to monitor convergence of diameters
for i in range(100):
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    flow_balance.check_flow_balance()
    distensibility.update_vessel_diameters()
    print("Distensibility update: it=" + str(i + 1) + ", residual = " + "{:.2e}".format(
        np.max(np.abs(flow_network.diameter - diameters_current))) + " um (tol = " + "{:.2e}".format(tol)+")")
    if np.max(np.abs(flow_network.diameter - diameters_current)) < tol:
        print("Distensibility update: DONE")
        break
    else:
        diameters_current = flow_network.diameter
print("Update the diameters based on Distensibility Law: DONE")

flow_network.write_network()
