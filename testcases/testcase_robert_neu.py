from source.flow_network import FlowNetwork
from source.inverse_model import InverseModel
from types import MappingProxyType
import source.setup.setup as setup
# todo: in setup class



# todo: read dict from file; need a good way to import from human readable file.
# todo: Problem: Json does not support comments; need better solution...

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
        "rbc_impact_option": 2,  # 1: No RBCs (hd=0)
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: todo Other laws. in vivo?
        "solver_option": 1,  # 1: Direct solver
                             # 2-...: other solvers (CG, AMG, ...)
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
        "hexa_boundary_types": [1, 1],
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
        # Inverse problem options
        # Define parameter space
        "parameter_space": 1,  # 1: Relative diameter to baseline (alpha = d/d_base)
                               # todo 2, 3, ...
        "parameter_restriction": 2,  # 1: No restriction of parameter values (alpha_prime = alpha)
                                     # 2: Restriction of parameter by a +/- tolerance to baseline
        "inverse_model_solver": 1,  # Direct solver
        # Target edges
        "csv_path_edge_target_data": "data/inverse_model/edge_target.csv",
        # Parameter edges # todo: also vertex parameters, distinguish
        "csv_path_edge_parameterspace": "data/inverse_model/edge_parameters.csv",
        # Gradient descent options:
        "gamma": .5,
        "phi": .5
    }
)

setup_simulation = setup.SetupSimulation()

imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver = setup_simulation.setup_simulation(PARAMETERS)

imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter, imp_adjoint_solver, \
    imp_alpha_mapping = setup_simulation.setup_inverse_model(PARAMETERS)

# build flownetwork and  object
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem, imp_solver, imp_velocity, PARAMETERS)
inverse_model = InverseModel(flow_network, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter, imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

print("Read network: ...")
flow_network.read_network()
print("Read network: DONE")

print("Update transmissibility: ...")
flow_network.update_transmissibility()
print("Update transmissibility: DONE")

print("Update flow, pressure and velocity: ...")
flow_network.update_blood_flow()
print("Update flow, pressure and velocity: DONE")

flow_network.write_network()

inverse_model.initialise_inverse_model()

for i in range(100):
    inverse_model.update_state()
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()

print(flow_network.flow_rate[inverse_model.edge_constraint_eid])
print(flow_network.rbc_velocity[inverse_model.edge_constraint_eid])
print(inverse_model.alpha)