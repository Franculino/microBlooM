"""
A python script to estimate edge parameters such as diameters and transmissibilities of microvascular networks based
on given flow rates and velocities in selected edges. Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Update the diameters and transmissibilities with a gradient descent algorithm minimising a given cost function.
5. Restriction of parameters to desired ranges (target value +/- tolerance).
6. Individual selection of parameter edges and target edges.
7. Target flow rates and velocities can be specified and combined into a single cost function.
8. Tuning of either relative diameters or relative transmissibilities compared to baseline.
9. Optimisation of diameters for a fixed number of iteration steps.
10. Save the results in a file.
"""

from source.flow_network import FlowNetwork
from source.inverse_model import InverseModel
from types import MappingProxyType
import source.setup.setup as setup


# MappingProxyType is basically a const dict
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 1,  # 1: generate hexagonal graph
                                   # 2: import graph from csv files
                                   # 3: todo import graph from igraph files
                                   # 4: todo import graph from edge_data and vertex_data pickle files
        "write_network_option": 2,  # 1: do not write anything
                                    # 2: igraph format
                                    # 3-...: todo other file formats. also handle overwrite data, etc
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
                                       # 3: todo: RBC tracking
                                       # 4-xxx: todo: steady state RBC laws
        "rbc_impact_option": 2,  # 1: hd = ht (makes only sense if tube_haematocrit_option:1, with ht=0)
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: todo Other laws. in vivo?
        "solver_option": 1,  # 1: Direct solver
                             # 2: PyAMG solver
                             # 3-...: other solvers (CG, AMG, ...)

        # Blood properties
        "ht_constant": 0.3,
        "mu_plasma": 0.0012,
        # Hexagonal network properties
        "nr_of_hexagon_x": 3,
        "nr_of_hexagon_y": 3,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 27],
        "hexa_boundary_values": [2, 1],
        "hexa_boundary_types": [1, 1],  # 1: pressure, 2: flow rate
        # Import network from csv options
        "csv_path_vertex_data": "data/network/b6_B_pre_061/node_data.csv",
        "csv_path_edge_data": "data/network/b6_B_pre_061/edge_data.csv",
        "csv_path_boundary_data": "data/network/b6_B_pre_061/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "boundaryValue",
        # Write options
        "write_override_initial_graph": False,
        "write_path_igraph": "data/network/b6_B_pre_061_simulated.pkl", # only required for "write_network_option" 2
        ##########################
        # Inverse problem options
        # Define parameter space
        "parameter_space": 1,  # 1: Relative diameter to baseline (alpha = d/d_base)
                               # 2: Relative transmissibility to baseline (alpha = T/T_base)
        "parameter_restriction": 2,  # 1: No restriction of parameter values (alpha_prime = alpha)
                                     # 2: Restriction of parameter by a +/- tolerance to baseline
        "inverse_model_solver": 1,  # 1: Direct solver
                                    # 2: PyAMG solver
                                    # 3-...: other solvers
        # Target edges
        "csv_path_edge_target_data": "data/inverse_model/edge_target.csv",
        # Parameter edges
        "csv_path_edge_parameterspace": "data/inverse_model/edge_parameters.csv",
        # Gradient descent options:
        "gamma": .5,
        "phi": .5,
        "max_nr_of_iterations": 50
    }
)

setup_simulation = setup.SetupSimulation()

# Initialise objects related to simulate blood flow without RBC tracking.
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver = setup_simulation.setup_bloodflow_model(PARAMETERS)

# Initialise objects related to the inverse model.
imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter, imp_adjoint_solver, \
    imp_alpha_mapping = setup_simulation.setup_inverse_model(PARAMETERS)

# Initialise flownetwork and inverse model objects
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)
inverse_model = InverseModel(flow_network, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                             imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

print("Read network: ...")
flow_network.read_network()
print("Read network: DONE")

print("Update transmissibility: ...")
flow_network.update_transmissibility()
print("Update transmissibility: DONE")

print("Update flow, pressure and velocity: ...")
flow_network.update_blood_flow()
print("Update flow, pressure and velocity: DONE")

inverse_model.initialise_inverse_model()
inverse_model.update_cost()

nr_of_iterations = int(PARAMETERS["max_nr_of_iterations"])
print("Solve the inverse problem and update the diameters: ...")
for i in range(nr_of_iterations):
    inverse_model.update_state()
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    inverse_model.update_cost()

    if i % 5 == 0:
        print(str(i)+" / " + str(nr_of_iterations) + " iterations done (f_H =", "%.2e" % inverse_model.f_h+")")
print(str(nr_of_iterations-1)+" / " + str(nr_of_iterations) + " iterations done (f_H =", "%.2e" % inverse_model.f_h+")")
print("Solve the inverse problem and update the diameters: DONE")

flow_network.write_network()

print("Type\t\tEid\t\tVal_tar_min\t\tVal_tar_max,\tVal_opt,\tVal_base ")
for eid, value, range, type in zip(inverse_model.edge_constraint_eid, inverse_model.edge_constraint_value, inverse_model.edge_constraint_range_pm, inverse_model.edge_constraint_type):
    if type==1:
        print("Flow rate","\t",eid,"\t",value-range,"\t","\t",value+range,"\t","\t","%.2e" % flow_network.flow_rate[eid])
    elif type==2:
        print("Velocity","\t",eid,"\t",value-range,"\t",value+range,"\t","\t","\t", "%.2e" % flow_network.rbc_velocity[eid])
