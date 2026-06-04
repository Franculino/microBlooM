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
4. Update the vessel diameters based on our distensibility and autoregulation models
5. Save the results in a file
"""
import sys
import numpy as np
import igraph
import matplotlib.pyplot as plt

from source.flow_network import FlowNetwork
from source.distensibility import Distensibility
from source.autoregulation import Autoregulation
from types import MappingProxyType
import source.setup.setup as setup

import time
import concurrent.futures


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
        "rbc_impact_option": 3,  # 1: No RBCs (hd=0)
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: Laws by Pries and Secomb (2005)
        "solver_option": 1,  # 1: Direct solver
                             # 2: PyAMG solver
                             # 3: Pardiso solver
                             # 4-...: other solvers
        "iterative_routine": 1,  # 1: Forward problem
                                 # 2: Iterative routine (ours)
                                 # 3: Iterative routine (Berg Thesis) [https://oatao.univ-toulouse.fr/25471/1/Berg_Maxime.pdf]
                                 # 4: Iterative routine (Rasmussen et al. 2018) [https://onlinelibrary.wiley.com/doi/10.1111/micc.12445]

        # Elastic vessel - vascular properties (tube law) - Only required for distensibility and autoregulation models
        "pressure_external": 0.,                    # Constant external pressure
        "read_vascular_properties_option": 2,       # 1: Do not read anything
                                                    # 2: Read vascular properties from csv file
        "tube_law_ref_state_option": 4,             # 1: No compute of reference diameters (d_ref)
                                                    # 2: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = p_base,
                                                        # d_ref = d_base
                                                    # 3: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const,
                                                        # d_ref computed based on Sherwin et al. (2003)
                                                    # 4: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const,
                                                        # d_ref computed based on Payne et al. (2023)
        # Vascular properties - edge properties:
        # - eid: edge ids
        # - e_modulus: Young's modulus of the vessel
        # - wall_thickness: Wall thickness of the vessel
        "csv_path_vascular_properties": "data/vascular_properties/all_eids_vascular_properties.csv",

        # Blood properties
        "ht_constant": 0.3,  # only required if RBC impact is considered
        "mu_plasma": 0.0012,

        # Zero Flow Vessel Threshold
        # True: the vessel with low flow are set to zero
        # The threshold is set as the max of mass-flow balance
        # The function is reported in set_low_flow_threshold()
        "ZeroFlowThreshold": False,

        # Hexagonal network properties. Only required for "read_network_option" 1
        "nr_of_hexagon_x": 3,
        "nr_of_hexagon_y": 3,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 27],
        "hexa_boundary_values": [2, 1],
        "hexa_boundary_types": [1, 1],  # 1: pressure & 2: flow rate

        # Import network from csv options. Only required for "read_network_option" 2
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
        "write_path_igraph": "data/network/network_simulated",

        ##########################
        # Vessel distensibility options
        ##########################

        # Set up distensibility model
        "read_dist_parameters_option": 2,       # 1: Do not read anything
                                                # 2: Read from csv file

        "dist_pres_area_relation_option": 2,    # 1: No update of diameters due to vessel distensibility
                                                # 2: Relation based on Sherwin et al. (2003) - non linear p-A relation

        # Distensibility edge parameters:
        # - eid_distensibility: edge ids for distensible vessels
        "csv_path_distensibility": "data/distensibility/distensibility_parameters.csv",

        ##########################
        # Autoregulation options
        ##########################

        # Modelling constants
        "sensitivity_direct_stress": 4.,        # Sensitivity factor of direct stresses
        "sensitivity_shear_stress": .5,         # Sensitivity factor of direct stresses

        "relaxation_factor": 0.1,               # Alpha - relaxation factor

        # Set up distensibility model
        "read_auto_parameters_option": 2,       # 1: Do not read anything
                                                # 2: Read from csv file

        "base_compliance_relation_option": 2,   # 1: Do not specify compliance relation
                                                # 2: Baseline compliance using the definition C = dV/dPt based on Sherwin et al. (2023)

        "auto_feedback_model_option": 2,        # 1: No update of diameters due to autoregulation
                                                # 2: Our approach - Update diameters by adjusting the autoregulation model proposed by Payne et al. (2023)

        # Autoregulation edge parameters:
        # - eid_autoregulation: edge ids for autoregulatory vessels
        # - rel_sensitivity_direct_stress: relative senistivity factor for direct stresses (this variable will be multiplied by "sensitivity_direct_stress" in the read_autoregulation_parameters.py)
        # - rel_sensitivity_shear_stress: relative senistivity factor for shear stresses (this variable will be multiplied by "sensitivity_shear_stress" in the read_autoregulation_parameters.py)
        "csv_path_autoregulation": "data/autoregulation/autoregulation_parameters.csv",

    }
)

def model_simulation(percent):

    # Create object to set up the simulation and initialise the simulation
    setup_simulation = setup.SetupSimulation()
    # Initialise the implementations based on the parameters specified
    imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
        imp_solver, imp_iterative, imp_balance, imp_read_vascular_properties, imp_tube_law_ref_state = setup_simulation.setup_bloodflow_model(PARAMETERS)

    imp_read_dist_parameters, imp_dist_pres_area_relation = setup_simulation.setup_distensibility_model(PARAMETERS)

    imp_read_auto_parameters, imp_auto_baseline, imp_auto_feedback_model = setup_simulation.setup_autoregulation_model(PARAMETERS)

    # Build flownetwork object and pass the implementations of the different submodules, which were selected in
    #  the parameter file
    flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                               imp_solver, imp_velocity, imp_iterative, imp_balance, imp_read_vascular_properties,
                               imp_tube_law_ref_state, PARAMETERS)

    distensibility = Distensibility(flow_network, imp_read_dist_parameters, imp_dist_pres_area_relation)

    autoregulation = Autoregulation(flow_network, imp_read_auto_parameters, imp_auto_baseline, imp_auto_feedback_model)

    flow_network.percent_pressure_change = round(percent * 100)

    # Import or generate the network - Import data for the pre-stroke state
    print("Read network: ...")
    flow_network.read_network()
    print("Read network: DONE")

    # Baseline
    # Diameters at baseline.
    # They are needed to compute the reference pressure and diameters
    print("Solve baseline flow (for reference): ...")
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    print("Solve baseline flow (for reference): DONE")

    print("Check flow balance: ...")
    flow_network.check_flow_balance()
    print("Check flow balance: DONE")

    if percent == 1:
        flow_network.write_network()
        return 1.

    print("Initialise tube law for elastic vessels based on baseline results: ...")
    flow_network.initialise_tube_law()
    print("Initialise tube law for elastic vessels based on baseline results: Done")

    # Save pressure filed and diameters at baseline.
    autoregulation.diameter_baseline = np.copy(flow_network.diameter)
    autoregulation.pressure_baseline = np.copy(flow_network.pressure)
    autoregulation.flow_rate_baseline = np.copy(flow_network.flow_rate)

    flow_network.diameter_baseline = np.copy(flow_network.diameter)

    print("Initialise distensibility model based on baseline results: ...")
    distensibility.initialise_distensibility()
    print("Initialise distensibility model based on baseline results: DONE")

    print("Initialise autoregulation model: ...")
    autoregulation.initialise_autoregulation()
    autoregulation.alpha = PARAMETERS["relaxation_factor"]
    print("Initialise autoregulation model: DONE")

    ### Modify this part based on your simulation scenario ###
    # Change the intel pressure boundary condition - Mean arterial pressure (MAP) of the network
    print("Change the intel pressure boundary condition - MAP: ...")
    flow_network.boundary_val[0] *= percent  # change the inlet pressure -- a 15 % drop in the starting inlet pressure
    print("Change the intel pressure boundary condition - MAP: DONE")

    print("Autogulation Region - Autoregulatory vessels change their diameters based on Compliance feedback model", flush=True)
    # Update diameters and iterate (has to be improved)
    print("Update the diameters based on Compliance feedback model: ...", flush=True)
    tol = 1.0E-6
    autoregulation.diameter_previous = flow_network.diameter  # Previous diameters to monitor convergence of diameters
    max_rel_change_ar = np.array([])
    end_iteration = 0
    max_iterations = 2000000
    for i in range(max_iterations):
        autoregulation.iteration = i
        flow_network.update_transmissibility()
        flow_network.update_blood_flow()
        flow_network.check_flow_balance()
        distensibility.update_vessel_diameters_dist()
        autoregulation.update_vessel_diameters_auto()
        rel_change = np.abs(
            (flow_network.diameter - autoregulation.diameter_previous) / autoregulation.diameter_previous)
        max_rel_change_ar = np.append(max_rel_change_ar, np.max(rel_change))
        # convergence criteria
        if (i + 1) % 10 == 0:
            print("Percent: "+str(round(percent * 100))+"% - Autoregulation update: it=" + str(i + 1) +
                  ", residual = " + "{:.2e}".format(np.max(rel_change)) + " (tol = " + "{:.2e}".format(tol) + ")")

        if np.max(rel_change) < tol:
            print("Autoregulation update: DONE")
            end_iteration = i
            break
        else:
            autoregulation.diameter_previous = np.copy(flow_network.diameter)
            if i == max_iterations - 1:
                sys.exit("Fail to update the diameters based on Compliance feedback model ...")
    print("Update the diameters based on Compliance feedback model: DONE")

    fig, ax = plt.subplots()
    itarations = np.arange(1, end_iteration + 2, dtype=int)
    ax.plot(itarations, max_rel_change_ar)
    ax.set_yscale('log')
    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel("Max Rel. Diameter Change [-]", fontsize=16)
    ax.set_title("Convergence Curve", fontsize=16)
    plt.savefig("output/simulation_monitoring_autoregulation/Convergence_curve_" + str(flow_network.percent_pressure_change) + ".png")
    plt.close()

    # Export data
    rel_stiffness = autoregulation.rel_stiffness
    rel_compliance = autoregulation.rel_compliance
    sensitivity_shear = autoregulation.sens_shear
    sensitivity_direct = autoregulation.sens_direct

    flow_network.rel_stiffness = np.ones(flow_network.nr_of_es) * (-1.)
    flow_network.rel_stiffness[autoregulation.eid_vessel_autoregulation] = rel_stiffness
    flow_network.rel_compliance = np.ones(flow_network.nr_of_es) * (-1.)
    flow_network.rel_compliance[autoregulation.eid_vessel_autoregulation] = rel_compliance
    flow_network.sensitivity_shear = np.zeros(flow_network.nr_of_es)
    flow_network.sensitivity_shear[autoregulation.eid_vessel_autoregulation] = sensitivity_shear
    flow_network.sensitivity_direct = np.zeros(flow_network.nr_of_es)
    flow_network.sensitivity_direct[autoregulation.eid_vessel_autoregulation] = sensitivity_direct

    flow_network.write_network()

    return


# Function to execute in parallel
def task(percent):
    print("\nPercent of Inlet Pressure Chance: " + str(round(percent * 100)) + "%")
    model_simulation(percent)
    return


print("[Network Name]")

# Number of CPUs to use - Adjust
# num_cpus = multiprocessing.cpu_count()
num_cpus = 20

# List of items to process
inlet_percent = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

# Record the start time
start_time = time.time()

# Using ThreadPoolExecutor with 20 workers
with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
    # Submit tasks and wait for them to complete
    futures = [executor.submit(task, percent) for percent in inlet_percent]
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        # print(f"\nResult from iteration {result}")

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"All tasks completed in {elapsed_time:.2f} seconds.", flush=True)

print("#########")
