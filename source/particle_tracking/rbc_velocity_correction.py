import sys
import numpy as np
import pandas as pd
import random
import itertools
import igraph as ig
import matplotlib.pyplot as plt
import os
import vtk
import csv
import igraph

from source.flow_network import FlowNetwork
from source.bloodflowmodel.flow_balance import FlowBalance
from types import MappingProxyType
import source.setup.setup as setup

class Particle_tracker(object):

    def __init__(self, PARAMETERS: MappingProxyType, flow_network: FlowNetwork):
        self.flow_network = flow_network
        self._PARAMETERS = PARAMETERS
        # Retrieve the user-chosen output directory
        self.output_dir = self._PARAMETERS.get("output_directory", "data/network/outputs")

        self.es = flow_network.edge_list
        self.vs_coords = flow_network.xyz
        self.length = flow_network.length
        self.diameter = flow_network.diameter
        self.flow_rate = flow_network.flow_rate
        self.rbc_velocity = flow_network.rbc_velocity
        self.bulk_velocity = self.flow_rate / (np.square(self.diameter) * np.pi / 4)
        self.pressure = flow_network.pressure
        self.volume = self.get_volumes()
        self.flow_network.volume = self.volume
        self.max_particles_vessel = np.floor(self.volume / self.flow_network.rbc_volume).astype(int)
        
        self.graph = ig.Graph()  # Initialize the graph
        self.graph.add_vertices(self.vs_coords.shape[0])  # Add vertices
        self.graph.add_edges(self.es)  # Add edges

        # Asign attributes to vertices and edges
        self.graph.vs['xyz'] = self.vs_coords.tolist()  # Coordinates of the vertices
        self.graph.vs['pressure'] = self.pressure # Pressure of the verrtices
        self.graph.es['length'] = self.length  # Length of the edges
        self.graph.es['diameter'] = self.diameter  # Diameter of the edges
        self.graph.es['flow_rate'] = self.flow_rate  # Flow rate through the edges
        self.graph.es['rbc_velocity'] = self.rbc_velocity # Rbc_velocity
        
        self.use_tortuosity = PARAMETERS["use_tortuosity"]
        self.parallel = PARAMETERS['parallel']
        self.times_basic_delta_t = self._PARAMETERS['times_basic_delta_t']
        self.delta_t = self.times_basic_delta_t * abs(self.length).min()/(abs(self.rbc_velocity).max())
        self.N_timesteps =  self._PARAMETERS["N_timesteps"]
        self.out_particles = []

        num_vessels = len(self.flow_network.edge_list)
        self.hematocrit_evolution = np.zeros((num_vessels, self.N_timesteps))  # Shape: (vessels, timesteps)
        self.num_particles_evolution = np.zeros((num_vessels, self.N_timesteps))  # Shape: (vessels, timesteps)
        self.volume_evolution = np.zeros((num_vessels, self.N_timesteps)) 

        if self.use_tortuosity == 1:
            graph2 = igraph.Graph.Read_Pickle(self._PARAMETERS['pkl_path_igraph'])
            self.vessel_data = {}

            self.points = graph2.es["points"]
            self.lengths = graph2.es["lengths2"]

            # check if the points (subnodes) and the nodes of the edges are defined in the same direction.
            for edge in self.graph.es:
                source, target = edge.tuple
                coords_source = self.graph.vs[source]['xyz']
                coords_target = self.graph.vs[target]['xyz']
                edge_points = self.points[edge.index]

                if not (np.allclose(coords_source, edge_points[0]) and np.allclose(coords_target, edge_points[-1])):
                    self.points[edge.index] = self.points[edge.index][::-1]
                    self.lengths[edge.index] = self.lengths[edge.index][::-1]

        else:

            self.points = None
            self.lengths = None
    
        # for computing the hematocrit
        self.flow_network.num_particles_in_vessel = np.zeros(len(self.flow_network.edge_list), dtype=int)
        self.ht_initial = PARAMETERS["ht_initial"]
        self.ht_boundary_condition = PARAMETERS["ht_boundary_condition"]
        self.rbc_volume = PARAMETERS["rbc_volume"]

        # Velocity sign change in vessels
        self.vessels_direction_changes = np.zeros(len(self.rbc_velocity), dtype=int)
       
        self.boundary_vertices = self.flow_network.boundary_vs
        self.boundary_vessels = set()
        for bv in self.boundary_vertices:
                    for edge_id in self.graph.incident(bv, mode="ALL"):
                        self.boundary_vessels.add(edge_id)
        self.boundary_vessels = list(self.boundary_vessels)
        self.inflow_vertices, self.outflow_vertices = self.detect_inflow_outflow_vertices()
        self.inflow_vertices = np.array(self.inflow_vertices)
        self.node_classification = self.classify_nodes(self.graph)
        self.inflow_vessels, self.outflow_vessels= self.detect_possible_inflow_outflow_vessels() 
       
        self.initialize_particles_with_hematocrit2() 
        
        self.ghost_particles = self.initialization_ghost_vessels()
        self.total_added_particles = 0

    def initialize_particles_with_hematocrit2(self):
        self.initial_particles_per_vessel = np.zeros(len(self.es), dtype=int)
        for vessel_id in range(len(self.es)):
            self.initial_particles_per_vessel[vessel_id] = int((self.volume[vessel_id] * self.ht_initial) // self.rbc_volume)
        
        self.N_particles = sum(self.initial_particles_per_vessel)
        # Total number of initialized particles
        self.N_particles_total = int(self.N_particles + 300)
        self.N_particles_count = int(self.N_particles)
        print('Total number of initialized particles:', self.N_particles_total)

        initial_vessels = []
        initial_local_coords = []
        for vessel_id in range(len(self.initial_particles_per_vessel)):
            num_particles_in_vessel = self.initial_particles_per_vessel[vessel_id]
            
            if num_particles_in_vessel > 0:
                initial_vessels.extend([vessel_id] * num_particles_in_vessel)
                diameter = self.diameter[vessel_id]
                min_distance = 1.1 * (self.rbc_volume / (np.pi * (diameter / 2)**2))
                possible_positions = np.arange(0.01, 0.99, min_distance / self.length[vessel_id])
                if len(possible_positions) < num_particles_in_vessel:
                    raise ValueError(
                        f"Not enough space in vessel {vessel_id} for {num_particles_in_vessel} particles "
                        f"with minimum distance {min_distance}. Reduce the particle density or increase the vessel length."
                    )
                coords = sorted(np.random.choice(possible_positions, num_particles_in_vessel, replace=False))
                initial_local_coords.extend(coords)
        self.particles_evolution = np.zeros((self.N_particles_total, self.N_timesteps + 1, 2), dtype=object)
        self.initial_position = np.array([[int(tube), coord] for tube, coord in zip(initial_vessels, initial_local_coords)])
        self.particles_evolution[:self.N_particles, 0, :] = self.initial_position
        self.particles_evolution[self.N_particles:, :, :] = np.nan
        self.inactive_particles = np.zeros(self.N_particles_total, dtype=bool)
        for vessel in initial_vessels:
            self.flow_network.num_particles_in_vessel[vessel] += 1

        previous_rbc_velocity = self.rbc_velocity.copy()
        self.update_network()
        prev_boundary_vel = self.get_boundary_velocities(previous_rbc_velocity)
        curr_boundary_vel = self.get_boundary_velocities(self.rbc_velocity)
        changed_vessels_boundary = self.detect_boundary_velocity_sign_change(prev_boundary_vel,
                                                                            curr_boundary_vel)
        self.update_boundary_classification_after_sign_change(changed_vessels_boundary)
     
    def detect_inflow_outflow_vertices(self):

        inflow_vertices = []
        outflow_vertices = []

        for bv in self.boundary_vertices:
            edges_connected = self.graph.incident(bv, mode="ALL")
            for edge_id in edges_connected:
                source, target = self.es[edge_id]
                other = target if source == bv else source
                if self.pressure[bv] > self.pressure[other]:
                    inflow_vertices.append(bv)
                else:
                    outflow_vertices.append(bv)

        return inflow_vertices, outflow_vertices

    def get_volumes(self):

            '''
            Get volumes of all the vessels in the network (cylindric approximation).
            '''

            volume = self.length * np.pi * self.diameter**2 / 4 
            return volume

    def expand_arrays_if_needed(self):
        """Expand the array if the next timestep the size won't be enough"""
        if self.N_particles_count >= self.N_particles_total:
            # Aumentar el número total de partículas en 3000
            self.N_particles_total += 5000
            
            # Expansión de particles_evolution
            new_particles_evolution = np.zeros((self.N_particles_total, self.N_timesteps + 1, 2), dtype=object)
            new_particles_evolution[:self.particles_evolution.shape[0], :, :] = self.particles_evolution 
            new_particles_evolution[self.particles_evolution.shape[0]:, :, :] = np.nan
            
            # Expansión de inactive_particles
            new_inactive_particles = np.zeros(self.N_particles_total, dtype=bool)
            new_inactive_particles[:self.inactive_particles.shape[0]] = self.inactive_particles
            
            # Expansión de particle_size
            new_particle_size = np.zeros(self.N_particles_total)
            new_particle_size[:self.particle_size.shape[0]] = self.particle_size

            # Asignar los nuevos arrays expandidos
            self.particles_evolution = new_particles_evolution
            self.inactive_particles = new_inactive_particles
            self.particle_size = new_particle_size
            
            print("The arrays have been updated: ", self.N_particles_count, self.N_particles_total, self.particle_size)

    def evolve_particles(self):
        """Evolve particles across each timestep. Computes the movement of every particles in the network"""
        self.particle_size = np.zeros(self.particles_evolution.shape[0])
        previous_rbc_velocity = self.rbc_velocity.copy()
        # print('Timestep: ', self.delta_t)

        for t in range(1, self.N_timesteps + 1):
            self.delta_t = self.times_basic_delta_t * abs(self.length).min()/(abs(self.rbc_velocity).max())
            print('Delta_t = ', self.delta_t)

            for vessel_idx in range(len(self.flow_network.edge_list)):
                self.hematocrit_evolution[vessel_idx, t-1] = self.flow_network.ht[vessel_idx]
                self.num_particles_evolution[vessel_idx, t-1] = self.flow_network.num_particles_in_vessel[vessel_idx]
                self.volume_evolution[vessel_idx, t-1] = self.volume[vessel_idx]
                
            # Determine active particles for this timestep
            current_timestep_particles = self.particles_evolution[:self.N_particles_count, t-1, 0].astype(float)
            active_particles = np.where(~np.isnan(current_timestep_particles))[0]

            # Calculate the total distance to travel for all particles
            initial_vessels = self.particles_evolution[active_particles, t-1, 0].astype(int)
            alpha_old        = self.particles_evolution[active_particles, t-1, 1]  # Local position in [0,1]
            length_array     = self.length[initial_vessels]
            velocity_array   = self.rbc_velocity[initial_vessels]
            distance_array   = velocity_array * self.delta_t 

            diameters = self.diameter[initial_vessels]
            self.particle_size[active_particles] = self.rbc_volume / (np.pi * diameters**2 / 4)
            
            alpha_new = alpha_old + (distance_array / length_array)

            inside_mask = (alpha_new >= 0.0) & (alpha_new <= 1.0)
            left_mask   = (alpha_new < 0.0)
            right_mask  = (alpha_new > 1.0)

            # 1) Particles that remain in the same vessel
            same_vessel_particles = active_particles[inside_mask]
            self.particles_evolution[same_vessel_particles, t, 0] = initial_vessels[inside_mask]
            self.particles_evolution[same_vessel_particles, t, 1] = alpha_new[inside_mask]

            crossing_right = active_particles[right_mask]
            if len(crossing_right) > 0:
                old_vessels_right = initial_vessels[right_mask]
                alpha_old_right   = alpha_old[right_mask]
                dist_travel_right = distance_array[right_mask]

                time_to_1 = (1.0 - alpha_old_right) * (length_array[right_mask] / velocity_array[right_mask])
                leftover_time = self.delta_t - time_to_1
                node_1 = self.flow_network.edge_list[old_vessels_right, 1]

                for i, particle_idx in enumerate(crossing_right):
                    old_vessel = old_vessels_right[i]
                    node       = node_1[i]
                    leftover   = leftover_time[i]

                    new_vessel = self.select_next_vessel(old_vessel, node)
                    if new_vessel is None:
                        # Sale de la red: marcamos NaN y la añadimos a out_particles
                        self.out_particles.append(particle_idx)
                        self.particles_evolution[particle_idx, t:, :] = np.nan
                        self.inactive_particles[particle_idx] = True
                        self.flow_network.num_particles_in_vessel[old_vessel] -= 1
                    elif new_vessel != old_vessel:
                        self.flow_network.num_particles_in_vessel[new_vessel] += 1
                        self.flow_network.num_particles_in_vessel[old_vessel] -= 1
                        vel_new     = self.rbc_velocity[new_vessel]
                        length_new  = self.length[new_vessel]
                        
                        alpha_start = 0.0  if vel_new >= 0 else 1.0
                        alpha_step  = (vel_new * leftover) / length_new
                        alpha_final = alpha_start + alpha_step
                        if (alpha_final> 1.0) or (alpha_final< 0.0):
                            print("WARNING: A particle is not being propagated correctly: you should decrease the timestep (deecrease times_basic_delta_t)")

                        self.particles_evolution[particle_idx, t, 0] = new_vessel
                        self.particles_evolution[particle_idx, t, 1] = alpha_final
                    else:
                        # No space for the particle in the possible vessels
                        self.particles_evolution[particle_idx, t, 0] = old_vessel
                        self.particles_evolution[particle_idx, t, 1] = 1.0

            crossing_left = active_particles[left_mask]
            if len(crossing_left) > 0:
                old_vessels_left = initial_vessels[left_mask]
                alpha_old_left   = alpha_old[left_mask]
                dist_travel_left = distance_array[left_mask]

                time_to_0 = (alpha_old_left - 0.0) * (length_array[left_mask] / abs(velocity_array[left_mask]))
                leftover_time = self.delta_t - time_to_0

                node_0 = self.flow_network.edge_list[old_vessels_left, 0]

                for i, particle_idx in enumerate(crossing_left):
                    old_vessel = old_vessels_left[i]
                    node       = node_0[i]
                    leftover   = leftover_time[i]
                    new_vessel = self.select_next_vessel(old_vessel, node)
                    if new_vessel is None:
                        self.out_particles.append(particle_idx)
                        self.particles_evolution[particle_idx, t:, :] = np.nan
                        self.inactive_particles[particle_idx] = True
                        self.flow_network.num_particles_in_vessel[old_vessel] -= 1
                    elif new_vessel != old_vessel:
                        self.flow_network.num_particles_in_vessel[new_vessel] += 1
                        self.flow_network.num_particles_in_vessel[old_vessel] -= 1

                        # Avanzar en el nuevo vaso con leftover
                        vel_new     = self.rbc_velocity[new_vessel]
                        length_new  = self.length[new_vessel]
                        alpha_start = 0.0 if vel_new >= 0 else 1.0
                        alpha_step  = (vel_new * leftover) / length_new
                        alpha_final = alpha_start + alpha_step
                        if (alpha_final> 1.0) or (alpha_final< 0.0):
                            print("WARNING: A particle is not being propagated correctly: you should decrease the timestep (deecrease times_basic_delta_t)")

                        self.particles_evolution[particle_idx, t, 0] = new_vessel
                        self.particles_evolution[particle_idx, t, 1] = alpha_final
                    else:
                        # No space for the particle in the possible vessels
                        self.particles_evolution[particle_idx, t, 0] = old_vessel
                        self.particles_evolution[particle_idx, t, 1] = 0.0
                    
            remaining_capacity = self.max_particles_vessel[self.inflow_vessels] - self.flow_network.num_particles_in_vessel[self.inflow_vessels]
            inflow_particles, number_inflowing_particles = self.update_ghost_particles(self.ghost_particles, self.inflow_vessels, remaining_capacity)
            self.N_particles_count = int(self.N_particles_count + number_inflowing_particles)
            self.expand_arrays_if_needed()

            if number_inflowing_particles > 0:
                idx_start = self.N_particles_count - number_inflowing_particles
                idx_end   = self.N_particles_count
                self.particles_evolution[idx_start:idx_end, t, 0] = inflow_particles[:, 0]
                self.particles_evolution[idx_start:idx_end, t, 1] = inflow_particles[:, 1]

            previous_rbc_velocity = self.rbc_velocity.copy()
            self.update_network()
            sign_change_indices = self.detect_velocity_sign_change(previous_rbc_velocity, self.rbc_velocity)
            prev_boundary_vel = self.get_boundary_velocities(previous_rbc_velocity)
            curr_boundary_vel = self.get_boundary_velocities(self.rbc_velocity)
            changed_vessels_boundary = self.detect_boundary_velocity_sign_change(prev_boundary_vel,
                                                                             curr_boundary_vel)
            self.update_boundary_classification_after_sign_change(changed_vessels_boundary)
            print('Timesetp: ', t)
        
        total_changes = np.sum(self.vessels_direction_changes)
        percentage_changed = (total_changes / len(self.vessels_direction_changes)) * 100
        print(f"Total vessels that changed direction during the simulation: {total_changes}/{len(self.vessels_direction_changes)}")
        print(f"Percentage of vessels that changed direction: {percentage_changed:.4f}%")
        
    
    def select_next_vessel(self, old_vessel, crossed_node):
        """
        Determines the next vessel the particle moves to after crossing the 
        node 'crossed_node' from 'old_vessel'. Returns None if it's an outflow 
        node (particle exits the network).

            - The starting node (crossed_node) must have a higher pressure 
            than the node the edge leads to (to ensure flow).
        """
        # 1) If 'crossed_node' is an outflow node, the particle exits the network
        if crossed_node in self.outflow_vertices:
            return None

        # 2) Edges connected to 'crossed_node' (in mode ALL to inspect both directions)
        connected_edges = self.graph.incident(crossed_node, mode="ALL")

        # Avoid returning to the same vessel the particle came from:
        connected_edges = [e for e in connected_edges if e != old_vessel]

        valid_edges = []
        for e in connected_edges:
            n0, n1 = self.es[e]  # nodes that form edge e

            # If starting from node n0, the flow must be n0 -> n1, i.e., P[n0] > P[n1]
            if crossed_node == n0 and self.pressure[n0] > self.pressure[n1]:
                # Also check if the vessel e has remaining capacity
                if (self.max_particles_vessel[e] -
                    self.flow_network.num_particles_in_vessel[e]) > 0:
                    valid_edges.append(e)

            # If starting from node n1, the flow must be n1 -> n0, i.e., P[n1] > P[n0]
            elif crossed_node == n1 and self.pressure[n1] > self.pressure[n0]:
                if (self.max_particles_vessel[e] -
                    self.flow_network.num_particles_in_vessel[e]) > 0:
                    valid_edges.append(e)

        # 3) If no valid edges exist, the particle "gets stuck" in the old_vessel
        if len(valid_edges) == 0:
            return old_vessel

        # 4) If there is only one valid edge, that is the new vessel
        if len(valid_edges) == 1:
            new_vessel = valid_edges[0]
        else:
            # If there are multiple valid edges, apply your bifurcation or selection function
            new_vessel = self.rbc_bifurcations(old_vessel, valid_edges)
            
        return new_vessel

    def rbc_bifurcations(self, old_vessel, valid_edges):

            if len(valid_edges) == 1:
                    new_vessel = valid_edges[0]

            elif len(valid_edges) == 2:
                total_flow_rate = sum(abs(self.flow_rate[e]) for e in valid_edges)
                FQ_B = [abs(self.flow_rate[e]) / total_flow_rate for e in valid_edges]
                FQ_B = np.array(FQ_B)
                D_f = self.diameter[old_vessel] * 10**6
                Hd = self.flow_network.hd[old_vessel]
                X_0 = 0.964 * (1-Hd) / D_f

                if FQ_B[0] <= X_0:
                    FQ_E = np.array([0,1])
                elif FQ_B[0] >= 1 - X_0:
                    FQ_E = np.array([1,0])
                elif X_0 < FQ_B[0] < 1 - X_0:
                    D_alpha = self.diameter[valid_edges[0]] * 10**6
                    D_beta = self.diameter[valid_edges[1]] * 10**6
                    
                    D_ratio = D_alpha**2 / D_beta**2
                    D_ratio_inverse = D_ratio**(-1)

                    A = np.zeros(2)
                    FQ_E = np.zeros(2)
                    probabilities = np.zeros(2)
                    internal_logit = np.zeros(2)
                    term = np.zeros(2)
                    A[0] = -13.29 * ((D_ratio - 1) / (D_ratio + 1)) * (1 - Hd) / D_f
                    A[1] = -13.29 * ((D_ratio_inverse - 1) / (D_ratio_inverse + 1)) * (1 - Hd) / D_f
                    B = 1 + 6.98 * (1-Hd) / D_f
                    
                    internal_logit[0] = self.logit((FQ_B[0] - X_0) / (1 - 2*X_0))
                    internal_logit[1] = self.logit((FQ_B[1] - X_0) / (1 - 2*X_0))
                    term[0] = A[0] + B * internal_logit[0]
                    term[1] = A[1] + B * internal_logit[1]
                    exp_term = np.exp(term)
                    
                    FQ_E = exp_term / (1 + exp_term)
                selected_edge = random.choices(valid_edges, weights=FQ_E, k=1)[0]
                new_vessel = selected_edge

            elif len(valid_edges) > 2:
                
                total_flow_rate = sum(abs(self.flow_rate[e]) for e in valid_edges)
                probabilities = [abs(self.flow_rate[e]) / total_flow_rate for e in valid_edges]

                selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]
                new_vessel = selected_edge

            else:
                print("Invalid number of bifurcating vessels.")
                new_vessel = None

            return new_vessel

    def logit(self, x):
            return np.log(x / (1 - x))

    def update_network(self):
            self.flow_network.update_transmissibility()
            self.flow_network.update_blood_flow()
            self.ht = self.flow_network.ht
            self.flow_rate = self.flow_network.flow_rate
            self.rbc_velocity = self.flow_network.rbc_velocity
            self.pressure = self.flow_network.pressure
            self.bulk_velocity = self.flow_rate / (np.square(self.diameter) * np.pi / 4)
            for i, valor in enumerate(self.ht):
                if valor > 1:
                    print(f"El valor de ht en la posición {i} es mayor que uno: {valor}")
                elif valor < 0:
                    print(f"El valor de ht en la posición {i} es negativo: {valor}")

    def detect_possible_inflow_outflow_vessels(self):
        inflow_vessels = []
        outflow_vessels = []

        for edge_id in self.boundary_vessels:
            s, t = self.es[edge_id]
            if s in self.boundary_vertices:
                if self.pressure[s] > self.pressure[t]:
                    inflow_vessels.append(edge_id)
                else:
                    outflow_vessels.append(edge_id)
            if t in self.boundary_vertices:
                if self.pressure[t] > self.pressure[s]:
                    inflow_vessels.append(edge_id)
                else:
                    outflow_vessels.append(edge_id)

        return inflow_vessels, outflow_vessels
    
    def initialization_ghost_vessels(self):
        """
        Initializes ghost vessels and generates particles for inflow vessels.

        Returns:
        - ghost_particles: Dictionary mapping inflow vessel IDs to particle positions.
        """
        ghost_particles = {}  # To store particle positions for each inflow vessel
        k = 300 # help to define length of ghost vessels
        for vessel_id in self.boundary_vessels:
            # Get properties of the inflow vessel
            diameter = self.diameter[vessel_id]
            bulk_velocity = self.bulk_velocity[vessel_id]
            
            # Define ghost vessel properties
            ghost_length = k * abs(bulk_velocity) * self.delta_t  # Length proportional to flow
            ghost_volume = np.pi * (diameter / 2)**2 * ghost_length  # Cylindrical volume
            
            # Calculate number of particles required
            num_particles = int((ghost_volume * self.ht_boundary_condition) // self.rbc_volume)
            min_distance = 1.1 * (self.rbc_volume / (np.pi * (diameter / 2)**2))
            
            # Generate all valid positions within the ghost vessel
            # Positions are spaced by min_distance, normalized to [0, 1] (relative positions)
            possible_positions = np.arange(0, 1, min_distance / ghost_length)

            # Ensure that the number of possible positions is greater than or equal to required particles
            if len(possible_positions) < num_particles:
                raise ValueError(
                    f"Not enough space in the ghost vessel for {num_particles} particles with minimum distance {min_distance}. "
                    f"Reduce the particle density or increase the ghost vessel length."
                )

        
            selected_positions = sorted(np.random.choice(possible_positions, num_particles, replace=False))
            # Store the positions in the dictionary
            ghost_particles[vessel_id] = {
                "positions":selected_positions,
                "ghost_length": ghost_length,
                "ghost_volume": ghost_volume,
                "number_particles": num_particles,
                "current_index": 0,
                "queue": 0,
            }

        return ghost_particles

    def update_ghost_particles(self, ghost_particles, active_vessels, remaining_capacity):
        """
        Updates the ghost particle positions for active inflow vessels based on the distance traveled.
        
        Now, stores a simple list of vessel ID and local position for each particle.

        Parameters:
        - ghost_particles: Dictionary containing ghost vessel particle data.
        - active_vessels: List of currently active inflow vessels.
        - timestep_volume: Dictionary with the volume of blood introduced per vessel.

        Returns:
        - inflow_particles: List of tuples, each containing the vessel ID and local position for each particle.
        """
        inflow_particles = []
        timestep_particles_count = 0
        k = 1000

        for active_vessel_idx, vessel_id in enumerate(active_vessels):
            # Get the ghost vessel data
            ghost_data = ghost_particles[vessel_id]
            positions = ghost_data["positions"]
            current_position = ghost_data["current_index"]  # This will track the position along the vessel
            queue_particles = ghost_data["queue"]  # Particles waiting to enter
            remaining_capacity_vessel = remaining_capacity[active_vessel_idx]
            # Get the properties of the ghost vessel
            diameter = self.diameter[vessel_id]
            bulk_velocity = self.bulk_velocity[vessel_id]

            # Calculate the length that should be "filled" with particles in this timestep
            # The total distance the RBCs would travel in the timestep
            distance_to_fill = abs(bulk_velocity) * self.delta_t  # distance to fill in this timestep

            # Total length of the ghost vessel
            ghost_length = ghost_data["ghost_length"]

            # If the total length filled exceeds the ghost vessel length, reset
            if (current_position + distance_to_fill) / ghost_length > 1:
                # Reset the current position since the particles have traversed the entire ghost vessel length
                ghost_data["current_index"] = 0.0  # Start position at the beginning
                current_position = 0  # Reset current position for the next timestep
                
                # Recalculate ghost vessel properties based on updated RBC velocity
                new_ghost_length = k * abs(bulk_velocity) * self.delta_t  # Compute the new length of the ghost vessel
                new_ghost_volume = np.pi * (diameter / 2)**2 * new_ghost_length  # Calculate the volume of the ghost vessel cylinder
                new_num_particles = int((new_ghost_volume * self.ht_boundary_condition) // self.rbc_volume)  # Determine the required number of particles
                min_distance = 1.1 * (self.rbc_volume / (np.pi * (diameter / 2)**2))

                possible_positions = np.arange(0, 1, min_distance / new_ghost_length)

                # Ensure that the number of possible positions is greater than or equal to required particles
                if len(possible_positions) < new_num_particles:
                    raise ValueError(
                        f"Not enough space in the ghost vessel for {new_num_particles} particles with minimum distance {min_distance}. "
                        f"Reduce the particle density or increase the ghost vessel length."
                    )

                # Randomly select the required number of particles from possible positions
                selected_positions = sorted(np.random.choice(possible_positions, new_num_particles, replace=False))
                # Update the ghost vessel's properties in the dictionary
                ghost_data["ghost_length"] = new_ghost_length  # Assign the new length
                ghost_data["number_particles"] = new_num_particles  # Assign the recalculated number of particles
                ghost_data["ghost_volume"] = new_ghost_volume  # Assign the recalculated number of particles
                ghost_data["positions"] = selected_positions  # Generate new random particle positions
                positions = ghost_data["positions"]
                ghost_length = ghost_data["ghost_length"]
            ghost_data["current_index"] += distance_to_fill  # Update position

            # Now, we need to determine which particles fall within the range of the distance filled
            start_position = current_position / ghost_length
            end_position = (current_position + distance_to_fill) / ghost_length
            positions = np.array(positions)
            # Get particles that are between start_position and end_position
            particles_in_range = positions[(positions >= start_position) & (positions < end_position)]
            # total_particles = len(particles_in_range) + queue_particles


            if len(particles_in_range) > remaining_capacity_vessel:
                # Select only the first `remaining_capacity_vessel` particles
                num_to_accept = remaining_capacity_vessel
                accepted_particles = np.sort(particles_in_range)[:num_to_accept]
                excess_particles = len(particles_in_range) - remaining_capacity_vessel

                # Update the queue with the excess particles
                ghost_data["queue"] = excess_particles

                min_distance = 1.1 * (self.rbc_volume / (np.pi * (diameter / 2)**2))
                min_distance_norm = min_distance / ghost_length

                # Place the excess particles near the edge for the next timestep
                new_queue_positions = []
                for i in range(excess_particles):
                    pos_candidate = end_position - (i+1)*min_distance_norm
                    if pos_candidate < 0.0:
                        break
                    new_queue_positions.append(pos_candidate)
                new_positions = np.concatenate([ghost_data["positions"], new_queue_positions])
                ghost_data["positions"] = np.sort(new_positions)
            else:
                # If all particles can fit, reset the queue and accept all particles
                accepted_particles = particles_in_range
                ghost_data["queue"] = 0

            # Append the accepted particles to inflow_particles
            for pos_ghost in accepted_particles:
                alpha_magnitude = (end_position - pos_ghost) * (ghost_length / self.length[vessel_id])

                if bulk_velocity >= 0.0:
                    # Velocidad > 0 => particle goes from 0 to 1
                    alpha_local = alpha_magnitude
                else:
                    # Vel < 0 => particles goes from 1 to 0
                    alpha_local = 1.0 - alpha_magnitude
                alpha_local = max(0.0, min(1.0, alpha_local))
                inflow_particles.append([vessel_id, alpha_local])

            self.flow_network.num_particles_in_vessel[vessel_id] += len(accepted_particles)
            timestep_particles_count += len(accepted_particles)

        self.total_added_particles += timestep_particles_count
        print(timestep_particles_count)
        print(self.total_added_particles)
        return np.array(inflow_particles), timestep_particles_count

    def create_vtk_particles_per_timestep(self,particles_evolution_global, output_dir):
        num_particles, num_timesteps, _ = particles_evolution_global.shape
        
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create an index file for multiple timesteps
        index_file = os.path.join(output_dir, 'particles_timestep_index.vtk')
        with open(index_file, 'w') as f:
            f.write('# vtk DataFile Version 3.0\n')
            f.write('Particles Timesteps Index\n')
            f.write('ASCII\n')
            f.write('DATASET COLLECTION\n')
            for t in range(num_timesteps):
                timestep_file = f'particles_timestep_{t}.vtk'
                f.write(f'DATASET {timestep_file}\n')

        for t in range(num_timesteps):
            # Collect all valid particle positions for the current timestep in a NumPy array
            valid_mask = ~np.isnan(particles_evolution_global[:, t, 0])
            valid_positions = particles_evolution_global[valid_mask, t, :]
            
            if valid_positions.size == 0:
                continue  # Skip if no valid positions for the timestep

            # Create vtkPolyData object for the current timestep
            polydata = vtk.vtkPolyData()
            
            # Create vtkPoints and populate it with valid_positions
            points = vtk.vtkPoints()
            for pos in valid_positions:
                points.InsertNextPoint(pos)
            polydata.SetPoints(points)
            
            # Create and assign the timestep scalar
            scalars = vtk.vtkFloatArray()
            scalars.SetName("Timestep")
            scalars.SetNumberOfValues(len(valid_positions))
            for i in range(len(valid_positions)):
                scalars.SetValue(i, t)
            polydata.GetPointData().SetScalars(scalars)
            
            # Create the VTK writer
            timestep_file = os.path.join(output_dir, f'particles_timestep_{t}.vtk')
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(timestep_file)
            writer.SetInputData(polydata)
            writer.Write()

    def transform_to_global_coordinates(self, parallel=False):
        """
        Transforms the local coordinates of the particles to global coordinates.
        If `parallel` is set to True, the computation will be distributed across MPI processes.
        """

        # PARALLEL IMPLEMENTATION
        if self.parallel == True:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            particles_per_process = self.N_particles_total // size
            start_particle = rank * particles_per_process
            end_particle = (rank + 1) * particles_per_process if rank != size - 1 else self.N_particles_total
            local_particles_count = end_particle - start_particle

            particles_evolution_global_local = np.full((local_particles_count, self.N_timesteps + 1, 3), np.nan)
            
            if self.use_tortuosity == 1:
                if rank == 0:

                    for vessel_id in range(len(self.es)):
                        vessel_points = np.array(self.points[vessel_id])
                        vessel_lengths = np.array(self.lengths[vessel_id])
                        vessel_total_length = self.length[vessel_id]

                        normalized_lengths = np.cumsum(vessel_lengths) / vessel_total_length
                        normalized_lengths = np.insert(normalized_lengths, 0, 0)

                        self.vessel_data[vessel_id] = {
                            'points': vessel_points,
                            'normalized_lengths': normalized_lengths
                        }
                self.vessel_data = comm.bcast(self.vessel_data, root=0)
                comm.Barrier()
            
            for t in range(self.N_timesteps + 1):
                for p in range(start_particle, end_particle):
                    local_idx = p - start_particle  # Local index of the particle
                    vessel_id = self.particles_evolution[p, t, 0]
                    local_coord = self.particles_evolution[p, t, 1]

                    if np.isnan(vessel_id) or np.isnan(local_coord):
                        continue

                    vessel_id = int(vessel_id)

                    if self.use_tortuosity == 1:
                        vessel_points = self.vessel_data[vessel_id]['points']
                        normalized_lengths = self.vessel_data[vessel_id]['normalized_lengths']
                        
                        point_idx = np.searchsorted(normalized_lengths, local_coord, side='right') - 1
                        point_idx = min(point_idx, len(vessel_points) - 2)

                        point_start = vessel_points[point_idx]
                        point_end = vessel_points[point_idx + 1]

                        local_start = normalized_lengths[point_idx]
                        local_end = normalized_lengths[point_idx + 1]

                        interpolation_factor = (local_coord - local_start) / (local_end - local_start)
                        particle_global_position = point_start + interpolation_factor * (point_end - point_start)

                    elif self.use_tortuosity == 0:
                       
                        start_vertex = self.es[vessel_id, 0]
                        end_vertex = self.es[vessel_id, 1]
                        start_coords = self.vs_coords[start_vertex]
                        end_coords = self.vs_coords[end_vertex]

                        direction_vector = end_coords - start_coords
                        particle_global_position = start_coords + local_coord * direction_vector

                    particles_evolution_global_local[local_idx, t] = particle_global_position

            particles_evolution_global = None
            if rank == 0:
                particles_evolution_global = np.zeros((self.N_particles_total, self.N_timesteps + 1, 3))

            sendcounts = np.array([particles_per_process] * size)
            sendcounts[-1] = self.N_particles_total - (size - 1) * particles_per_process
            displacements = np.array([i * particles_per_process for i in range(size)])

            comm.Gatherv(
                particles_evolution_global_local, 
                [particles_evolution_global, 
                sendcounts * (self.N_timesteps + 1) * 3, 
                displacements * (self.N_timesteps + 1) * 3, 
                MPI.DOUBLE], 
                root=0
            )
            if rank == 0:
                return particles_evolution_global
            else:
                return None  
        else:
            
            # SEQUENTIAL IMPLEMENTATION 

            particles_evolution_global = np.full((self.N_particles_total, self.N_timesteps + 1, 3), np.nan)
            self.vessel_data = {}

            if self.use_tortuosity == 1:
                for vessel_id in range(len(self.es)):
                    vessel_points = np.array(self.points[vessel_id])
                    vessel_lengths = np.array(self.lengths[vessel_id])
                    vessel_total_length = self.length[vessel_id]

                    normalized_lengths = np.cumsum(vessel_lengths) / vessel_total_length
                    normalized_lengths = np.insert(normalized_lengths, 0, 0)

                    self.vessel_data[vessel_id] = {
                        'points': vessel_points,
                        'normalized_lengths': normalized_lengths
                    }

                for p in range(self.N_particles_total):
                    for t in range(self.N_timesteps + 1):
                        vessel_id = self.particles_evolution[p, t, 0]
                        local_coord = self.particles_evolution[p, t, 1]

                        if np.isnan(vessel_id) or np.isnan(local_coord):
                            continue

                        vessel_id = int(vessel_id)

                        vessel_points = self.vessel_data[vessel_id]['points']
                        normalized_lengths = self.vessel_data[vessel_id]['normalized_lengths']

                        point_idx = np.searchsorted(normalized_lengths, local_coord, side='right') - 1
                        point_idx = min(point_idx, len(vessel_points) - 2)

                        point_start = vessel_points[point_idx]
                        point_end = vessel_points[point_idx + 1]

                        local_start = normalized_lengths[point_idx]
                        local_end = normalized_lengths[point_idx + 1]

                        interpolation_factor = (local_coord - local_start) / (local_end - local_start)
                        particle_global_position = point_start + interpolation_factor * (point_end - point_start)

                        particles_evolution_global[p, t] = particle_global_position
                # self.save_particles_evolution_global_to_excel(particles_evolution_global)   
                return particles_evolution_global

            elif self.use_tortuosity == 0:
                for p in range(self.N_particles_total):
                    for t in range(self.N_timesteps + 1):
                        vessel_id = self.particles_evolution[p, t, 0]
                        local_coord = self.particles_evolution[p, t, 1]

                        if np.isnan(vessel_id) or np.isnan(local_coord):
                            continue

                        vessel_id = int(vessel_id)
                        start_vertex = self.es[vessel_id, 0]
                        end_vertex = self.es[vessel_id, 1]
                        start_coords = self.vs_coords[start_vertex]
                        end_coords = self.vs_coords[end_vertex]

                        direction_vector = end_coords - start_coords
                        particle_global_position = start_coords + local_coord * direction_vector

                        particles_evolution_global[p, t] = particle_global_position
                return particles_evolution_global

            else:
                raise ValueError(f"Invalid use_tortuosity: {self.use_tortuosity}. It must be either 0 or 1.")
            
    def classify_nodes(self, graph):
        # 0: Inflow node.
        # 1: Normal connection point.
        # 2: Convergent bifurcation.
        # 3: Divergent bifurcation.
        # 4: Outflow node.

        node_classification = np.zeros(graph.vcount(), dtype=int)

        node_classification[self.inflow_vertices] = 0  # Inflow nodes
        node_classification[self.outflow_vertices] = 4  # Outflow nodes

        for node in range(graph.vcount()):
            if node in self.inflow_vertices or node in self.outflow_vertices:
                continue  #
            
            connected_edges = graph.incident(node, mode="ALL")
            inflow_edges = []
            outflow_edges = []

            for edge_idx in connected_edges:
                edge = self.es[edge_idx]
                source, target = edge  # Orden fijo de los nodos
                if self.pressure[source] > self.pressure[target]:  # Flujo de source -> target
                    if target == node:
                        inflow_edges.append(edge_idx)
                    else:
                        outflow_edges.append(edge_idx)
                else:  # Flujo de target -> source
                    if source == node:
                        inflow_edges.append(edge_idx)
                    else:
                        outflow_edges.append(edge_idx)

            num_connected = len(connected_edges)

            if num_connected == 2:
                node_classification[node] = 1  # Normal connection point

            elif num_connected == 3:
                # Bifurcación convergente o divergente
                if len(inflow_edges) == 1 and len(outflow_edges) == 2:
                    node_classification[node] = 3  # Divergent bifurcation
                elif len(inflow_edges) == 2 and len(outflow_edges) == 1:
                    node_classification[node] = 2  # Convergent bifurcation
                else:
                    node_classification[node] = -1
                    


            elif num_connected >= 4:
                # Procesar bifurcaciones complejas
                if len(inflow_edges) == 3 and len(outflow_edges) == 1:
                    node_classification[node] = 2  # Convergent bifurcation
                elif len(inflow_edges) == 1 and len(outflow_edges) == 3:
                    node_classification[node] = 3  # Divergent bifurcation
                elif len(inflow_edges) == 2 and len(outflow_edges) == 2:
                    node_classification[node] = 3  # Divergent bifurcation
                else:
                    node_classification[node] = -1

        return node_classification
        
    def classify_boundary_node(self, bv):
        inflow_list = []
        outflow_list = []
        edges_connected = self.graph.incident(bv, mode="ALL")
        for edge_id in edges_connected:
            s, t = self.es[edge_id]
            other = t if s == bv else s
            
            if self.pressure[bv] > self.pressure[other]:
                inflow_list.append(bv)
            else:
                outflow_list.append(bv)
        return inflow_list, outflow_list
    
    def classify_boundary_vessel(self, edge_idx):
        inflow_list = []
        outflow_list = []
        s, t = self.es[edge_idx]
        if s in self.boundary_vertices:
            if self.pressure[s] > self.pressure[t]:
                inflow_list.append(edge_idx)
            else:
                outflow_list.append(edge_idx)
        if t in self.boundary_vertices:
            if self.pressure[t] > self.pressure[s]:
                inflow_list.append(edge_idx)
            else:
                outflow_list.append(edge_idx)

        return inflow_list, outflow_list
    
    def update_boundary_classification_after_sign_change(self, changed_vessels_boundary):

        if len(changed_vessels_boundary) == 0:
            return
        print("These boundary vessels changed direction of flow:", changed_vessels_boundary)
        inflow_vessels_set = set(self.inflow_vessels)
        outflow_vessels_set = set(self.outflow_vessels)
        inflow_vertices_set = set(self.inflow_vertices)
        outflow_vertices_set = set(self.outflow_vertices)

        for edge_idx in changed_vessels_boundary:
           
            inflow_vessels_set.discard(edge_idx)
            outflow_vessels_set.discard(edge_idx)

            new_inflow_list, new_outflow_list = self.classify_boundary_vessel(edge_idx)
            for iv in new_inflow_list:
                inflow_vessels_set.add(iv)    
            for ov in new_outflow_list:
                outflow_vessels_set.add(ov)

            s, t = self.es[edge_idx]
            boundary_nodes = []
            if s in self.boundary_vertices:
                boundary_nodes.append(s)
            if t in self.boundary_vertices:
                boundary_nodes.append(t)

            for bv in boundary_nodes:
                inflow_vertices_set.discard(bv)
                outflow_vertices_set.discard(bv)
                new_inflow_verts, new_outflow_verts = self.classify_boundary_node(bv)
                for inv in new_inflow_verts:
                    inflow_vertices_set.add(inv)  
                for ovv in new_outflow_verts:
                    outflow_vertices_set.add(ovv)

        self.inflow_vessels = list(inflow_vessels_set)
        self.outflow_vessels = list(outflow_vessels_set)
        self.inflow_vertices = list(inflow_vertices_set)
        self.outflow_vertices = list(outflow_vertices_set)

        self.inflow_vessels.sort()
        self.outflow_vessels.sort()
        self.inflow_vertices.sort()
        self.outflow_vertices.sort()
        
    def get_boundary_velocities(self, velocities_full):
        return velocities_full[self.boundary_vessels]

    def detect_boundary_velocity_sign_change(self, prev_vel_boundary, curr_vel_boundary):
        sign_change_local = np.where(np.sign(prev_vel_boundary) != np.sign(curr_vel_boundary))[0]
        sign_change_global = [self.boundary_vessels[i] for i in sign_change_local]
        return sign_change_global
    
    def detect_velocity_sign_change(self, previous_velocities, current_velocities):
        sign_change_indices = np.where(np.sign(previous_velocities) != np.sign(current_velocities))[0]
        self.vessels_direction_changes[sign_change_indices] = 1
        return sign_change_indices