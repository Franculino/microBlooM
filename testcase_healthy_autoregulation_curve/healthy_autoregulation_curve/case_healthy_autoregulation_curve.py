"""
A python script to simulate stationary blood flow in microvascular networks with considering the vascular distensibility
and autoregulation mechanisms. In response to pressure perturbations (e.g., healthy conditions, ischaemic stroke), the
cerebral autoregulation feedback mechanisms act to change the wall stiffness (or the compliance), and hence the diameter,
of the autoregulatory microvessels.
Baseline is at healthy conditions for 100 and 10mmHg of the inlet and outlet boundary pressure, respectively.
The reference state for the distensibility law is computed based on the baseline condition.
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
from source.autoregulation import Autoregulation
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
        "write_network_option": 3,  # 1: do not write anything
                                    # 2: write to igraph format # todo: handle overwriting data from import file
                                    # 3: write to vtp format
                                    # 4: write to two csv files
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
                                       # 3: todo: RBC tracking
                                       # 4-...: todo: steady state RBC laws
        "rbc_impact_option": 3,  # 1: No RBCs (hd=0)
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
        "nr_of_hexagon_x": 3,
        "nr_of_hexagon_y": 3,
        "hexa_edge_length": 80e-6,
        "hexa_diameter": 18e-6,
        "hexa_boundary_vertices": [0, 27], # [0, 27]
        "hexa_boundary_values": [13332., 8665.],
        "hexa_boundary_types": [1, 1],
        "stroke_edges": [0, 1],  # Example: Occlude 2 edges at inflow - manually assigning of blocked vessel ids
        "diameter_blocked_edges": .5e-6,

        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "data/network/b6_B_pre_061/node_data.csv",
        "csv_path_edge_data": "data/network/b6_B_pre_061/edge_data.csv",
        "csv_path_boundary_data": "data/network/b6_B_pre_061/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "p",

        # Import network from igraph option. Only required for "read_network_option" 3
        "pkl_path_igraph": "data/network/B6_B_01/b6_B_pre_stroke.pkl",
        "ig_diameter": "diameter", "ig_length": "length", "ig_coord_xyz": "coords",
        "ig_boundary_type": "boundaryType",  # 1: pressure & 2: flow rate
        "ig_boundary_value": "boundaryValue",

        # Write options
        "write_override_initial_graph": False,  # todo: currently does not do anything
        "write_path_igraph": "testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/network/results_2version.vtp",  # only required for "write_network_option" 2, 3, 4


        ##########################
        # Vessel distensibility options
        ##########################

        # Set up distensibility model
        "read_dist_parameters_option": 2,       # 1: Do not read anything
                                                # 2: Read from csv file

        "dist_ref_state_option": 3,             # 1: No update of diameters due to vessel distensibility
                                                # 2: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = p_base,
                                                    # d_ref = d_base
                                                # 3: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const,
                                                    # d_ref computed based on Sherwin et al. (2003)
                                                # 4: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const,
                                                    # d_ref computed based on Urquiza et al. (2006)
                                                # 5: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const,
                                                    # d_ref computed based on Rammos et al. (1998)

        "dist_pres_area_relation_option": 2,    # 1: No update of diameters due to vessel distensibility
                                                # 2: Relation based on Sherwin et al. (2003) - non linear p-A relation
                                                # 3: Relation based on Urquiza et al. (2006) - non linear p-A relation
                                                # 4: Relation based on Rammos et al. (1998) - linear p-A relation

        # Distensibility edge properties
        "csv_path_distensibility": "testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/data/distensibility_parameters.csv",
        "pressure_external": 0.,  # Constant external pressure as reference pressure (only for distensibility_model 2)

        ##########################
        # Autoregulation options
        ##########################

        "MAP_mmHg": 85,     # Mean arterial pressure in mmHg

        "relaxation_factor": 0.2,               # Alpha - relaxation factor

        # Set up distensibility model
        "read_auto_parameters_option": 2,       # 1: Do not read anything
                                                # 2: Read from csv file

        "compliance_relation_option": 2,        # 1: Do not specify compliance relation
                                                # 2: compliance according to p-A relation proposed by Sherwin et al. (2003)

        "auto_feedback_model_option": 2,        # 1: No update of diameters due to autoregulation
                                                # 2: linear feedback model for the vessel relative stiffness based on Payne et al. (2023)
                                                # 3: linear feedback model for the vessel relative stiffness based on Ursino et al. (1997)
        # Autoregulation edge properties
        "csv_path_autoregulation": "testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/data/autoregulation_parameters.csv",
    }
)

get_plots = False

# print("\n MAP in mmHg:", PARAMETERS["MAP_mmHg"])
# percent = PARAMETERS["MAP_mmHg"]/100

MAP_array = np.arange(70., 182.5, 2.5)
percent_array = MAP_array/100
inflow = np.array([])
mean_diameter_array = np.array([])

inflow_base = None
diameter_base = None
for cur_MAP, percent in zip(MAP_array, percent_array):

    print("\n MAP in mmHg:", cur_MAP)

    # Create object to set up the simulation and initialise the simulation
    setup_blood_flow = setup.SetupSimulation()
    # Initialise the implementations based on the parameters specified
    imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
        imp_solver = setup_blood_flow.setup_bloodflow_model(PARAMETERS)

    imp_read_dist_parameters, imp_dist_ref_state, imp_dist_pres_area_relation = \
        setup_blood_flow.setup_distensibility_model(PARAMETERS)

    imp_read_auto_parameters, imp_auto_baseline, imp_auto_feedback_model = \
        setup_blood_flow.setup_autoregulation_model(PARAMETERS)

    # Build flownetwork object and pass the implementations of the different submodules, which were selected in
    #  the parameter file
    flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                               imp_solver, imp_velocity, PARAMETERS)

    flow_balance = FlowBalance(flow_network)

    distensibility = Distensibility(flow_network, imp_dist_ref_state, imp_read_dist_parameters,
                                    imp_dist_pres_area_relation)

    autoregulation = Autoregulation(flow_network, distensibility, imp_read_auto_parameters, imp_auto_baseline,
                                    imp_auto_feedback_model)

    # Import or generate the network - Import data for the pre-stroke state
    print("Read network: ...")
    flow_network.read_network()
    print("Read network: DONE")

    # Baseline
    # Diameters at baseline.
    # They are needed to compute the reference pressure and diameters - only for distensibility_ref_state: 3
    print("Solve baseline flow (for reference): ...")
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    print("Solve baseline flow (for reference): DONE")

    print("Check flow balance: ...")
    flow_balance.check_flow_balance()
    print("Check flow balance: DONE")

    print("Base inflow:", "{:.3e}".format(np.abs(flow_network.flow_rate[0]) + np.abs(flow_network.flow_rate[1])))

    # Save pressure filed and diameters at baseline.
    autoregulation.diameter_baseline = np.copy(flow_network.diameter)
    autoregulation.pressure_baseline = np.copy(flow_network.pressure)
    autoregulation.flow_rate_baseline = np.copy(flow_network.flow_rate)

    pressure_base = .5 * np.sum(np.copy(flow_network.pressure)[flow_network.edge_list], axis=1) / 133.3
    diameter_base = np.copy(flow_network.diameter)
    inflow_base = np.abs(flow_network.flow_rate[0]) + np.abs(flow_network.flow_rate[1])
    # inflow_base = np.abs(flow_network.flow_rate[0])

    print("Initialise distensibility model based on baseline results: ...")
    distensibility.initialise_distensibility()
    print("Initialise distensibility model based on baseline results: DONE")

    print("Initialise autoregulation model: ...")
    autoregulation.initialise_autoregulation()
    autoregulation.alpha = PARAMETERS["relaxation_factor"]
    print("Initialise autoregulation model: DONE")

    # Change the intel pressure boundary condition - Mean arterial pressure (MAP) of the network
    print("Change the intel pressure boundary condition - MAP: ...")
    bc_current = [13332. * percent, 8665.]  # change the inlet pressure
    print("bc_current", bc_current)
    flow_network.boundary_val = np.array(bc_current)
    print("Change the intel pressure boundary condition - MAP: DONE")

    # flow_rate_base = np.abs(np.copy(flow_network.flow_rate))

    if cur_MAP < 80 or cur_MAP > 150:
        # Update diameters and iterate (has to be improved)
        print("Update the diameters based on Distensibility Law: ...")
        tol = 1.e-10
        diameters_current = flow_network.diameter  # Previous diameters to monitor convergence of diameters
        for i in range(50):
            flow_network.update_transmissibility()
            flow_network.update_blood_flow()
            flow_balance.check_flow_balance()
            distensibility.update_vessel_diameters_dist()
            print("Distensibility update: it=" + str(i + 1) + ", residual = " + "{:.2e}".format(
                np.max(np.abs(flow_network.diameter - diameters_current))) + " um (tol = " + "{:.2e}".format(tol)+")")
            if np.max(np.abs(flow_network.diameter - diameters_current)) < tol:
                print("Distensibility update: DONE")
                break
            else:
                diameters_current = flow_network.diameter
        print("Update the diameters based on Distensibility Law: DONE")

    if cur_MAP >= 80 and cur_MAP <= 130:

        if cur_MAP > 100:
            autoregulation.alpha = 0.1

        print("Autogulation Region - Autoregulatory vessels change their diameters based on Compliance feedback model")
        # Update diameters and iterate (has to be improved)
        print("Update the diameters based on Compliance feedback model: ...")

        tol = 1.e-06
        autoregulation.diameter_previous = flow_network.diameter  # Previous diameters to monitor convergence of diameters

        diam_array = np.array([flow_network.diameter])
        rel_change_array = np.array([])
        max_rel_change_array = np.array([])
        rel_stiffness_array = np.array([])
        rel_compliance_array = np.array([])
        direct_stress_array = np.array([])
        shear_stress_array = np.array([])
        pressure_drop_array = np.array([])

        for i in range(1000):
            flow_network.update_transmissibility()
            flow_network.update_blood_flow()
            flow_balance.check_flow_balance()
            # distensibility.update_vessel_diameters_dist()
            autoregulation.update_vessel_diameters_auto()
            rel_change = np.abs((flow_network.diameter - autoregulation.diameter_previous)/autoregulation.diameter_previous)

            # plots
            if get_plots:
                rel_change_array = np.append(rel_change_array, rel_change)
                max_rel_change_array = np.append(max_rel_change_array, np.max(rel_change))
                diam_array = np.append(diam_array, flow_network.diameter)
                rel_stiffness_array = np.append(rel_stiffness_array, autoregulation.rel_stiffness)
                rel_compliance_array = np.append(rel_compliance_array, autoregulation.rel_compliance)
                direct_stress_array = np.append(direct_stress_array, autoregulation.direct_stress)
                shear_stress_array = np.append(shear_stress_array, autoregulation.shear_stress)
                pressure_drop_array = np.append(pressure_drop_array, autoregulation.pressure_drop_edge)

            # convergence criteria
            if (i + 1)%50 == 0:
                print("Autoregulation update: it=" + str(i + 1) + ", residual = " + "{:.2e}".format(np.max(rel_change)) + " um (tol = " + "{:.2e}".format(tol)+")")
            if np.max(rel_change) < tol:
                print("Autoregulation update: DONE")
                break
            else:
                autoregulation.diameter_previous = flow_network.diameter

    if cur_MAP > 130 and cur_MAP <= 150:

        # Update diameters and iterate (has to be improved)
        print("Update the diameters based on Distensibility Law: ...")
        tol = 1.e-10
        diameters_current = flow_network.diameter  # Previous diameters to monitor convergence of diameters
        for i in range(50):
            flow_network.update_transmissibility()
            flow_network.update_blood_flow()
            flow_balance.check_flow_balance()
            distensibility.update_vessel_diameters_dist()
            print("Distensibility update: it=" + str(i + 1) + ", residual = " + "{:.2e}".format(
                np.max(np.abs(flow_network.diameter - diameters_current))) + " um (tol = " + "{:.2e}".format(tol)+")")
            if np.max(np.abs(flow_network.diameter - diameters_current)) < tol:
                print("Distensibility update: DONE")
                break
            else:
                diameters_current = flow_network.diameter
        print("Update the diameters based on Distensibility Law: DONE")

        diameter_dis = np.copy(flow_network.diameter)

        flow_network.diameter = autoregulation.diameter_baseline
        bc_current = [13332. * 1.3, 8665.]  # change the inlet pressure
        print("bc_current", bc_current)
        flow_network.boundary_val = np.array(bc_current)

        if cur_MAP > 100:
            autoregulation.alpha = 0.1

        print("Autogulation Region - Autoregulatory vessels change their diameters based on Compliance feedback model")
        # Update diameters and iterate (has to be improved)
        print("Update the diameters based on Compliance feedback model: ...")

        tol = 1.e-06
        autoregulation.diameter_previous = flow_network.diameter  # Previous diameters to monitor convergence of diameters

        diam_array = np.array([flow_network.diameter])
        rel_change_array = np.array([])
        max_rel_change_array = np.array([])
        rel_stiffness_array = np.array([])
        rel_compliance_array = np.array([])
        direct_stress_array = np.array([])
        shear_stress_array = np.array([])
        pressure_drop_array = np.array([])

        for i in range(1000):
            flow_network.update_transmissibility()
            flow_network.update_blood_flow()
            flow_balance.check_flow_balance()
            # distensibility.update_vessel_diameters_dist()
            autoregulation.update_vessel_diameters_auto()
            rel_change = np.abs((flow_network.diameter - autoregulation.diameter_previous)/autoregulation.diameter_previous)

            # plots
            if get_plots:
                rel_change_array = np.append(rel_change_array, rel_change)
                max_rel_change_array = np.append(max_rel_change_array, np.max(rel_change))
                diam_array = np.append(diam_array, flow_network.diameter)
                rel_stiffness_array = np.append(rel_stiffness_array, autoregulation.rel_stiffness)
                rel_compliance_array = np.append(rel_compliance_array, autoregulation.rel_compliance)
                direct_stress_array = np.append(direct_stress_array, autoregulation.direct_stress)
                shear_stress_array = np.append(shear_stress_array, autoregulation.shear_stress)
                pressure_drop_array = np.append(pressure_drop_array, autoregulation.pressure_drop_edge)

            # convergence criteria
            if (i + 1)%50 == 0:
                print("Autoregulation update: it=" + str(i + 1) + ", residual = " + "{:.2e}".format(np.max(rel_change)) + " um (tol = " + "{:.2e}".format(tol)+")")
            if np.max(rel_change) < tol:
                print("Autoregulation update: DONE")
                break
            else:
                autoregulation.diameter_previous = flow_network.diameter

        diameter_auto = flow_network.diameter

        scaled_x = (cur_MAP - 130) / (150 - 130)
        coefficient = (np.tanh((scaled_x - 0.5) * 5) + 1) / 2

        flow_network.diameter = (1 - coefficient) * diameter_auto + coefficient * diameter_dis
        flow_network.update_transmissibility()
        flow_network.update_blood_flow()
        flow_balance.check_flow_balance()

        print("Update the diameters based on Compliance feedback model: DONE")

    print("Current inflow:", "{:.3e}".format(np.abs(flow_network.flow_rate[0]) + np.abs(flow_network.flow_rate[1])))
    cur_inflow = np.abs(flow_network.flow_rate[0]) + np.abs(flow_network.flow_rate[1])
    # cur_inflow = np.abs(flow_network.flow_rate[0])
    inflow = np.append(inflow, cur_inflow)

    cur_diameter = flow_network.diameter[0]
    mean_diameter_array = np.append(mean_diameter_array, np.mean(cur_diameter))

rel_inflow = inflow/inflow_base
# rel_mean_diameter = mean_diameter_array/ np.mean(diameter_base)
rel_mean_diameter = mean_diameter_array/ np.mean(diameter_base[0])

import matplotlib.pyplot as plt
plt.plot(MAP_array, rel_inflow, label="Inflow [m$^3$/s]")
# plt.plot(MAP_array, rel_mean_diameter, label="Mean Diameter [m]")
plt.plot(MAP_array, rel_mean_diameter, label="Diameter for Eid:0 [m]")
plt.legend()
plt.xlabel("MAP [mmHg]")
plt.ylabel("Relative value")
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/Inflow_Diam_vs_MAP.png")
plt.close()

sys.exit()
import matplotlib.pyplot as plt

iterations = np.arange(1, np.size(max_rel_change_array)+1, dtype=int)
plt.plot(iterations, max_rel_change_array)
plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Max Rel Diameter Change")
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/max_rel_diam_change_vs_iterations.png")
plt.close()

rel_change_array = rel_change_array.reshape((-1,autoregulation.nr_of_edge_autoregulation))
for i in range(autoregulation.nr_of_edge_autoregulation):
    plt.plot(iterations, rel_change_array[:,i])
plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Rel Diameter Change")
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/rel_diam_change_vs_iterations.png")
plt.close()

plt.plot(iterations, rel_change_array[:,0])
plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Rel Diameter Change")
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/rel_change_diam_vs_iterations_eid0.png")
plt.close()

plt.plot(iterations, rel_change_array[:,1])
plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Rel Diameter Change")
plt.yscale('log')
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/rel_change_diam_vs_iterations_eid1.png")
plt.close()

rel_stiffness_array = rel_stiffness_array.reshape((-1,autoregulation.nr_of_edge_autoregulation))
for i in range(autoregulation.nr_of_edge_autoregulation):
    plt.plot(iterations, rel_stiffness_array[:,i])
plt.axhline(y=1., linestyle="--", linewidth=2, color="black")
plt.xlabel("Iterations")
plt.ylabel("Relative Stiffness")
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/rel_stiffness_vs_iterations.png")
plt.close()

# for i in range(2):
#     plt.plot(iterations, rel_stiffness_array[:,i])
# plt.xlabel("Iterations")
# plt.ylabel("Relative Stiffness")
# plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/rel_stiffness_vs_iterations_0_1.png")
# plt.close()

rel_compliance_array = rel_compliance_array.reshape((-1,autoregulation.nr_of_edge_autoregulation))
for i in range(autoregulation.nr_of_edge_autoregulation):
    plt.plot(iterations, rel_compliance_array[:,i])
plt.axhline(y=1., linestyle="--", linewidth=2, color="black")
plt.xlabel("Iterations")
plt.ylabel("Relative Compliance")
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/rel_compliance_vs_iterations.png")
plt.close()

# for i in range(2):
#     plt.plot(iterations, rel_compliance_array[:,i])
# plt.xlabel("Iterations")
# plt.ylabel("Relative Compliance")
# plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/rel_compliance_vs_iterations_0_1.png")
# plt.close()

diam_array = diam_array.reshape((-1,flow_network.nr_of_es))
iter = np.arange(diam_array.shape[0], dtype=int)
for i in range(flow_network.nr_of_es):
    plt.plot(iter, diam_array[:,i])
plt.xlabel("Iterations")
plt.ylabel("Diameter [μm]")
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/diam_vs_iterations.png")
plt.close()

direct_stress_array = direct_stress_array.reshape((-1,autoregulation.nr_of_edge_autoregulation))
shear_stress_array = shear_stress_array.reshape((-1,autoregulation.nr_of_edge_autoregulation))
rel_direct_stress = direct_stress_array / autoregulation.direct_stress_baseline
rel_shear_stress = shear_stress_array / autoregulation.shear_stress_baseline
rel_diam_array = diam_array/autoregulation.diameter_baseline

plt.plot(iter[1:], rel_diam_array[1:, 0], label="Rel Diameter", color="blue")
plt.plot(iterations, rel_stiffness_array[:, 0], label="Rel Stiffness", color="red")
# plt.plot(iterations, rel_compliance_array[:, 0], label="Rel Compliance", color="green")
plt.plot(iterations, rel_direct_stress[:, 0], label="Rel Direct Stress", color="gold")
plt.plot(iterations, rel_shear_stress[:, 0], label="Rel Shear Stress", color="purple")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Relative Value")
# plt.ylim(0,5)
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/rel_values_vs_iterations_eid0.png")
plt.close()

plt.plot(iter[1:], rel_diam_array[1:, 1], label="Rel Diameter", color="blue")
plt.plot(iterations, rel_stiffness_array[:, 1], label="Rel Stiffness", color="red")
# plt.plot(iterations, rel_compliance_array[:, 1], label="Rel Compliance", color="green")
plt.plot(iterations, rel_direct_stress[:, 1], label="Rel Direct Stress", color="gold")
plt.plot(iterations, rel_shear_stress[:, 1], label="Rel Shear Stress", color="purple")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Relative Value")
# plt.ylim(0,5)
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/rel_values_vs_iterations_eid1.png")
plt.close()

pressure_drop_array = pressure_drop_array.reshape((-1,autoregulation.nr_of_edge_autoregulation)) / 133.3
re_pressure_drop_array = pressure_drop_array/pressure_base
plt.plot(iterations, re_pressure_drop_array[:,0])
plt.xlabel("Iterations")
plt.ylabel("Relative Pressure Drop")
plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/pressure_drop_vs_iterations.png")
plt.close()

print("Diameter - baseline:", autoregulation.diameter_baseline[0])
print("Diameter - after:", flow_network.diameter[0])
print("Pressure drop - baseline:", pressure_base[0])
print("Pressure drop - after:",pressure_drop_array[-1,0])
print("Pressure - baseline:", autoregulation.pressure_baseline[flow_network.edge_list][0])
print("Pressure - after:", flow_network.pressure[flow_network.edge_list][0])

# cur_inflow = np.abs(flow_network.flow_rate[0]) + np.abs(flow_network.flow_rate[1])
# inflow = np.append(inflow, cur_inflow)
# print("Current inflow:", "{:.3e}".format(cur_inflow))
#
# cur_diameter = flow_network.diameter
# mean_diameter_array = np.append(mean_diameter_array, np.mean(cur_diameter))
# rel_diameter = (cur_diameter - diameter_base)/ diameter_base
# print("Max/Min Rel Changes in Diameter:", "{:.3f}".format(np.max(rel_diameter)), "/", "{:.3f}".format(np.min(rel_diameter)))

# import matplotlib.pyplot as plt
# plt.plot(MAP_array, inflow)
# plt.xlabel("MAP [mmHg]")
# plt.ylabel("Inflow [m$^3$/s]")
# plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/Inflow_vs_MAP.png")
# plt.close()
# plt.plot(MAP_array, mean_diameter_array)
# plt.xlabel("MAP [mmHg]")
# plt.ylabel("Mean Diameter [m]")
# plt.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/MeanDiameter_vs_MAP.png")
# plt.close()


# flow_rate_bc_change = np.abs(flow_network.flow_rate)
# diameter_bc_change = np.copy(flow_network.diameter)
# pressure_bc_change = .5 * np.sum(flow_network.pressure[flow_network.edge_list], axis=1) / 133.3

from matplotlib.collections import LineCollection

 # Plot network
xy = flow_network.xyz[:, :2]
edgelist = flow_network.edge_list
segments = xy[edgelist]

# flow_rates_rel_change = (flow_rate_bc_change - flow_rate_base) / flow_rate_base
# print(flow_rates_rel_change)
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_xlim(xy[:, 0].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 0].max() + PARAMETERS['hexa_edge_length'])
# ax.set_ylim(xy[:, 1].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 1].max() + PARAMETERS['hexa_edge_length'])
# line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"),
#                                 norm=plt.Normalize(vmin=-1., vmax=0))
# line_segments.set_array(flow_rates_rel_change)
# line_segments.set_linewidth(diameter_bc_change * 1e5*2)
# ax.add_collection(line_segments)
# ax.set_title('Flow rate changes after MAP change, MAP = 120mmHg')
# ax.plot(xy[flow_network.boundary_vs, 0], xy[flow_network.boundary_vs, 1], 'o', color='r', markersize=10)
# cbar1 = plt.colorbar(line_segments)
# fig.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/flowrates_change.png", dpi=200)

# diameter_rel_change = (diameter_bc_change-diameter_base)/diameter_base
# print(diameter_rel_change)
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_xlim(xy[:, 0].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 0].max() + PARAMETERS['hexa_edge_length'])
# ax.set_ylim(xy[:, 1].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 1].max() + PARAMETERS['hexa_edge_length'])
# line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"),
#                                 norm=plt.Normalize(vmin=-1., vmax=0))
# line_segments.set_array(diameter_rel_change)
# line_segments.set_linewidth(diameter_bc_change * 1e5*2)
# ax.add_collection(line_segments)
# ax.set_title('Diameter changes after MAP change, MAP = 120mmHg')
# ax.plot(xy[flow_network.boundary_vs, 0], xy[flow_network.boundary_vs, 1], 'o', color='r', markersize=10)
# cbar1 = plt.colorbar(line_segments)
# fig.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/diameter_change.png", dpi=200)
pressure_bc_change = .5 * np.sum(np.copy(flow_network.pressure)[flow_network.edge_list], axis=1) / 133.3
diameter_bc_change = flow_network.diameter
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(xy[:, 0].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 0].max() + PARAMETERS['hexa_edge_length'])
ax.set_ylim(xy[:, 1].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 1].max() + PARAMETERS['hexa_edge_length'])
line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"),
                                norm=plt.Normalize(vmin=65, vmax=100))
line_segments.set_array(pressure_bc_change)
line_segments.set_linewidth(diameter_bc_change * 1e5*2)
ax.add_collection(line_segments)
ax.set_title('Pressure after MAP change, MAP = 85mmHg')
ax.plot(xy[flow_network.boundary_vs, 0], xy[flow_network.boundary_vs, 1], 'o', color='r', markersize=10)
cbar1 = plt.colorbar(line_segments)
fig.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/pressure.png", dpi=200)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(xy[:, 0].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 0].max() + PARAMETERS['hexa_edge_length'])
ax.set_ylim(xy[:, 1].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 1].max() + PARAMETERS['hexa_edge_length'])
line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"),
                                norm=plt.Normalize(vmin=65, vmax=100))
line_segments.set_array(pressure_base)
line_segments.set_linewidth(diameter_base * 1e5*2)
ax.add_collection(line_segments)
ax.set_title('Pressure baseline, MAP = 100mmHg')
ax.plot(xy[flow_network.boundary_vs, 0], xy[flow_network.boundary_vs, 1], 'o', color='r', markersize=10)
cbar1 = plt.colorbar(line_segments)
fig.savefig("testcase_healthy_autoregulation_curve/healthy_autoregulation_curve/output/pressure_base.png", dpi=200)

flow_network.write_network()
