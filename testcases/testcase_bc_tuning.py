"""
A python script to estimate vertex parameters such as boundary conditions of microvascular networks based
on given flow rates and velocities in selected edges.
Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Update the boundary pressures with a gradient descent algorithm minimising a given cost function.
5. Restriction of parameters to desired ranges (target value +/- tolerance).
6. Individual selection of parameter boundary vertices and target edges.
7. Target flow rates and velocities can be specified and combined into a single cost function.
8. Tuning of absolute boundary pressures.
9. Optimisation of pressures for a fixed number of iteration steps.
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
                                   # 3: import graph from igraph format (pickle file)
        "write_network_option": 2,  # 1: do not write anything
                                    # 2: write to igraph format (.pkl file)
                                    # 3: write to vtp format (.vtp)
                                    # 4: write to two csv files (.csv)
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
        "rbc_impact_option": 2,  # 1: No RBCs (hd=0) - makes only sense if tube_haematocrit_option:1 or ht=0
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: Laws by Pries and Secomb (2005)
        "solver_option": 1,  # 1: Direct solver
                             # 2: PyAMG solver

        # Blood properties
        "ht_constant": 0.3,
        "mu_plasma": 0.0012,

        # Hexagonal network properties
        "nr_of_hexagon_x": 5,
        "nr_of_hexagon_y": 5,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 65, 21, 48, 41, 31, 56],
        "hexa_boundary_values": [2, 1, 3, 1, 1, 5, 2],
        "hexa_boundary_types": [1, 1, 1, 1, 1, 1, 1],  # 1: pressure, 2: flow rate

        # Import network from csv options
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


        ##########################
        # Inverse problem options
        ##########################

        # Define parameter space
        "parameter_space": 11,  # 1: Relative diameter to baseline (alpha = d/d_base)
                                # 2: Relative transmissibility to baseline (alpha = T/T_base)
                                # 11: Pressure boundary condition values (alpha = p_0)
        "parameter_restriction": 1,  # 1: No restriction of parameter values (alpha_prime = alpha)
                                     # 2: Restriction of parameter by a +/- tolerance to baseline
        "inverse_model_solver": 1,  # Direct solver
                                    # 2: PyAMG solver

        # Filepath to prescribe target values / constraints on edges
        "csv_path_edge_target_data": "data/inverse_model/edge_target_BC_tuning.csv",
        # Filepath to define the edge parameter space (only for tuning of diameters and transmissibilities)
        "csv_path_edge_parameterspace": "not needed",
        # Filepath to define the vertex parameter space (only for tuning of boundary conditions)
        "csv_path_vertex_parameterspace": "data/inverse_model/vertex_parameters.csv",

        # Gradient descent options:
        "gamma": .5,
        "phi": .5,  # for parameter_restriction 2
        "max_nr_of_iterations": 50  # Maximum of iterations
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
cost_h = [inverse_model.f_h]
for i in range(nr_of_iterations):
    inverse_model.update_state()
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    inverse_model.update_cost()
    cost_h.append(inverse_model.f_h)

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

import matplotlib.pyplot as plt
plt.semilogy(cost_h)
plt.title("Cost")
plt.show()