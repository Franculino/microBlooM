import sys
import numpy as np
import pandas as pd
import random
import itertools
import igraph as ig
import matplotlib.pyplot as plt
import os
import vtk
import igraph

from source.flow_network import FlowNetwork
from source.bloodflowmodel.flow_balance import FlowBalance
from types import MappingProxyType
import source.setup.setup as setup

class Particle_tracker(object):

    def __init__(self, PARAMETERS: MappingProxyType, flow_network: FlowNetwork):
        self.flow_network = flow_network
        self._PARAMETERS = PARAMETERS

        self.es = flow_network.edge_list
        self.vs_coords = flow_network.xyz
        self.length = flow_network.length
        self.diameter = flow_network.diameter
        self.flow_rate = flow_network.flow_rate
        self.rbc_velocity = flow_network.rbc_velocity
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
        self.inflow_vertices, self.outflow_vertices = self.detect_inflow_outflow_vertices()
        self.inflow_vertices = np.array(self.inflow_vertices)
        self.out_particles = []
        self.particles_frequency = PARAMETERS["particles_frequency"]

        self.indices_rbc_negativa = np.where(self.rbc_velocity < 0)[0]
        self.es[self.indices_rbc_negativa] = self.es[self.indices_rbc_negativa][:, ::-1]

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

        self.intervals = self.get_intervals()

        # for computing the hematocrit
        self.flow_network.num_particles_in_vessel = np.zeros(len(self.flow_network.edge_list), dtype=int)
        self.ht_initial = PARAMETERS["ht_initial"]
        self.ht_boundary_condition = PARAMETERS["ht_boundary_condition"]
        self.rbc_volume = PARAMETERS["rbc_volume"]
        self.initial_particles_mode = PARAMETERS["initial_particles_mode"]

        if self.initial_particles_mode == 1:
            self.initialize_particles_with_hematocrit()
        elif self.initial_particles_mode == 0:
            self.N_particles = self._PARAMETERS["initial_number_particles"]
            self.initial_particle_tube = self._PARAMETERS["initial_vessels"]
            self.initial_particles_coords = np.zeros((self.N_particles, 3))
            self.initial_local_coord = np.full(self.N_particles, 0.5)
            self.initialize_particles_evolution() 

        self.inflow_vessels = self.detect_possible_inflow_vessels()
        self.ghost_particles = self.initialization_ghost_vessels()


    def initialize_particles_with_hematocrit(self):

            self.initial_particles_per_vessel = np.zeros(len(self.es), dtype = int)
            for vessel_id in range(len(self.es)):
                self.initial_particles_per_vessel[vessel_id] = int((self.volume[vessel_id] * self.ht_initial)  // self.rbc_volume)
            
            self.N_particles = sum(self.initial_particles_per_vessel)
            # Total number of initialized particles
            self.N_particles_total = int(self.N_particles + 3000)
            self.N_particles_count = int(self.N_particles)
            print('Total number of initialized particles:', self.N_particles_total)

            initial_vessels = []
            initial_local_coords = []
            for vessel_id in range(len(self.initial_particles_per_vessel)):
                num_particles_in_vessel = self.initial_particles_per_vessel[vessel_id]
                
                if num_particles_in_vessel > 0:
                    initial_vessels.extend([vessel_id] * num_particles_in_vessel)
                    if num_particles_in_vessel == 1:
                        coords = np.array([0.5]) 
                    else:
                        coords = 0.05 + (np.arange(1, num_particles_in_vessel + 1) / (num_particles_in_vessel + 1)) * 0.9
                    initial_local_coords.extend(coords)

            self.particles_evolution = np.zeros((self.N_particles_total, self.N_timesteps + 1, 2), dtype=object)
            self.initial_position = np.array([[int(tube), coord] for tube, coord in zip(initial_vessels, initial_local_coords)])
            self.particles_evolution[:self.N_particles, 0, :] = self.initial_position
            self.particles_evolution[self.N_particles:, :, :] = np.nan
            self.inactive_particles = np.zeros(self.N_particles_total, dtype=bool)
            for vessel in initial_vessels:
                self.flow_network.num_particles_in_vessel[vessel] += 1
            self.update_network()

    def detect_inflow_outflow_vertices(self):
            """
            Detect inflow and outflow vertices based on boundaryType and pressure values.
            """
            inflow_vertices = []
            outflow_vertices = []
            
            # Get the vertices that are on the boundary using the 'boundaryType' attribute
            boundary_vertices = [v.index for v in self.graph.vs if self._PARAMETERS['ig_boundary_type'] != 0]

            for bv in boundary_vertices:
                # Find all edges connected to the boundary vertex (bv)
                edges_connected = self.graph.incident(bv)

                for edge_id in edges_connected:
                    edge = self.graph.es[edge_id]
                    vertices = [edge.source, edge.target]

                    # Check if one of the vertices is the boundary vertex
                    if bv in vertices:
                        other_vertex = vertices[0] if vertices[1] == bv else vertices[1]

                        # Compare the pressures to determine if it's inflow or outflow
                        if self.graph.vs[bv]['pressure'] > self.graph.vs[other_vertex]['pressure']:
                            inflow_vertices.append(bv)
                        else:
                            outflow_vertices.append(bv)

            # Clean up vertices that are in both lists (inflow and outflow)
            inflow_vertices_set = set(inflow_vertices)
            outflow_vertices_set = set(outflow_vertices)
            vertices_in_both = inflow_vertices_set.intersection(outflow_vertices_set)

            inflow_vertices_clean = inflow_vertices_set - vertices_in_both
            outflow_vertices_clean = outflow_vertices_set - vertices_in_both

            return list(sorted(inflow_vertices_clean)), list(sorted(outflow_vertices_clean))

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
            self.N_particles_total += 3000
            
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
            
            print("The arrays have been updated: ", self.N_particles_count, self.N_particles_total)

    def evolve_particles(self):
            """Evolve particles across each timestep. Computes the movement of every particles in the net"""

            # print('Timestep: ', self.delta_t)
            for t in range(1, self.N_timesteps + 1):
                self.delta_t = self.times_basic_delta_t * abs(self.length).min()/(abs(self.rbc_velocity).max())
                print('Delta_t = ', self.delta_t)

                for vessel_idx in range(len(self.flow_network.edge_list)):
                    # Actualizar el hematocrito, número de partículas y volumen para cada vaso en este timestep
                    self.hematocrit_evolution[vessel_idx, t-1] = self.flow_network.ht[vessel_idx]
                    self.num_particles_evolution[vessel_idx, t-1] = self.flow_network.num_particles_in_vessel[vessel_idx]
                    self.volume_evolution[vessel_idx, t-1] = self.volume[vessel_idx]
                    
                # Determine active particles for this timestep
                active_particles_count = int(self.particles_per_timestep[t] - np.sum(self.inactive_particles[:self.particles_per_timestep[t]]))
                active_particles = np.where(~self.inactive_particles)[0]

                # Calculate the total distance to travel for all particles
                initial_vessels_per_iteration = self.particles_evolution[active_particles[:active_particles_count], t - 1, 0].astype(int)
                velocities_per_iteration = abs(self.rbc_velocity[initial_vessels_per_iteration])
                length_per_iteration = self.length[initial_vessels_per_iteration]
                distance_to_travel = velocities_per_iteration * self.delta_t

                # Calculate the remaining distance in the current vessels
                local_position_per_iteration = self.particles_evolution[active_particles[:active_particles_count], t - 1, 1]
                first_prediction_position = distance_to_travel / length_per_iteration + local_position_per_iteration
                change_vessel_positive_active_idx = np.where((first_prediction_position > 1))[0]
                change_vessel_positive = active_particles[change_vessel_positive_active_idx]
                same_vessel_active_idx = np.where(first_prediction_position <= 1)[0]
                same_vessel = active_particles[same_vessel_active_idx]

                # Particles that remain in the same vessel
                self.particles_evolution[same_vessel, t, 1] = first_prediction_position[same_vessel_active_idx]
                self.particles_evolution[same_vessel, t, 0] = initial_vessels_per_iteration[same_vessel_active_idx]

                # Particles that switch vessels
                remaining_time_positive = self.delta_t - (
                (1 - local_position_per_iteration[change_vessel_positive_active_idx]) 
                * length_per_iteration[change_vessel_positive_active_idx]
                ) / velocities_per_iteration[change_vessel_positive_active_idx]

                new_vessels, index_out_particles = self.select_vessels_positive(
                    initial_vessels_per_iteration[change_vessel_positive_active_idx], 
                    self.graph, 
                    self.outflow_vertices, 
                    self.flow_network.edge_list
                )
                
                new_vessels = new_vessels.astype(int)
                self.particles_evolution[change_vessel_positive, t, 0] = new_vessels
                staying_in_vessel_idx = np.where(new_vessels == initial_vessels_per_iteration[change_vessel_positive_active_idx])[0]
                self.particles_evolution[change_vessel_positive[staying_in_vessel_idx], t, 1] = 1.0
                moving_particles_idx = np.where(new_vessels != initial_vessels_per_iteration[change_vessel_positive_active_idx])[0]
                moving_particles = change_vessel_positive[moving_particles_idx]
                # THIS PART IS REMOVED FROM HERE AND INTRODUCED IN select_vessels_positive
                # old_vessels = self.particles_evolution[change_vessel_positive, t-1, 0].astype(int)  
                # for old_vessel, new_vessel in zip(old_vessels, new_vessels):
                #     self.flow_network.num_particles_in_vessel[old_vessel] -= 1 
                #     if new_vessel != 0:
                #         self.flow_network.num_particles_in_vessel[new_vessel] += 1
                
                # self.update_network()

                new_velocities = abs(self.rbc_velocity[new_vessels[moving_particles_idx]])

                second_prediction_position = new_velocities * remaining_time_positive[moving_particles_idx] / self.length[new_vessels[moving_particles_idx]]
                self.particles_evolution[moving_particles, t, 1] = second_prediction_position

                if np.any(second_prediction_position > 1):
                    print("A particles is not being propagated correctly: you should decrease the timestep")

                # Manage particles that exit the network.
                if index_out_particles:
                    for idx in index_out_particles:
                        outflow_vessel = int(self.particles_evolution[change_vessel_positive[idx], t - 1, 0])
                        # self.flow_network.num_particles_in_vessel[outflow_vessel] -= 1
                        self.out_particles.append(change_vessel_positive[idx])
                        self.particles_evolution[change_vessel_positive[idx], t - 1:, :] = np.nan
                        self.inactive_particles[change_vessel_positive[idx]] = True

                # Initialize particles entering the network in the next timestep
                # if self.particles_per_timestep[t + 1] > self.particles_per_timestep[t]:
                #     real_timestep = t + 1
                #     exact_divisors = [i for i in range(len(self.intervals)) if real_timestep % self.intervals[i] == 0]
                #     nodes_inflowing = self.inflow_vertices[exact_divisors]
                #     vessels_inflowing = self.select_vessels_inflow(nodes_inflowing, self.graph, self.flow_network.edge_list).astype(int)

                #     remaining_capacity = self.max_particles_vessel[vessels_inflowing] - self.flow_network.num_particles_in_vessel[vessels_inflowing]
                #     valid_vessels = vessels_inflowing[remaining_capacity > 1]
                #     valid_nodes = nodes_inflowing[remaining_capacity > 1]
                #     particles_not_injected = np.sum(remaining_capacity <= 1)

                #     if particles_not_injected > 0:
                #         self.particles_per_timestep[t + 1:] -= particles_not_injected

                #     number_inflowing_particles = self.particles_per_timestep[t + 1] - self.particles_per_timestep[t]
                #     self.particles_evolution[self.particles_per_timestep[t]:self.particles_per_timestep[t + 1], t, 0] = valid_vessels.astype(int)
                #     self.particles_evolution[self.particles_per_timestep[t]:self.particles_per_timestep[t + 1], t, 1] = 0.0
                #     for vessel in valid_vessels.astype(int):
                #         self.flow_network.num_particles_in_vessel[vessel] += 1
                # self.update_network()

            # Initialize particles entering the network in the next timestep
        
            # vessels_inflowing = self.select_vessels_inflow(self.inflow_vertices, self.graph, self.flow_network.edge_list).astype(int)
            remaining_capacity = int(self.max_particles_vessel[self.inflow_vessels] - self.flow_network.num_particles_in_vessel[self.inflow_vessels])

            inflow_particles, number_inflowing_particles = self.update_ghost_particles(self, self.ghost_particles, self.inflow_vessels, remaining_capacity):
            self.N_particles_count = int(self.N_particles_count + number_inflowing_particles)

            self.expand_arrays_if_needed()
            
            self.particles_evolution[int(self.N_particles_count - number_inflowing_particles): self.N_particles_count, t, 0] = inflow_particles[:][0]
            self.particles_evolution[int(self.N_particles_count - number_inflowing_particles): self.N_particles_count, t, 1] = inflow_particles[:][1]
            # for vessel in valid_vessels.astype(int):
            #     self.flow_network.num_particles_in_vessel[vessel] += 1
            self.update_network()
            print('Timesetp: ', t, 'Ht = :', self.flow_network.ht[0], '  Number of particles: ', self.flow_network.num_particles_in_vessel[0] )
            print('Ya acabo el bucle')
            # self.save_particles_evolution_to_excel()
            # self.save_vessel_data_to_excel()

    def select_vessels_positive(self, old_vessels, graph, outflow_vertices, es):
            """
            Selects the vessels into which the particles that change vessel in one timestep go.
            Checks if the end of the previous eddge is an outflow vertex.
            
            Parameters:
            - old_vessels: list of inflow vertices where particles will enter.
            - graph: the network graph representing the flow.
            - outflow_vertices: 
            - es: edge list representing the vessels in the network.
            
            Returns:
            - new_vessels: array of selected vessels into which the particles will inflow.
            - index_out_particles: particles that reach an ouflow vertex.
            """

            last_nodes = es[old_vessels, 1]
            new_vessels = np.zeros(len(old_vessels))
            index_out_particles = []
            support = 0
            
            for node in last_nodes:
                if node in outflow_vertices:
                    index_out_particles.append(support)
                    self.flow_network.num_particles_in_vessel[old_vessels[support]] -= 1
                    support += 1
                    continue  
                connected_edges = graph.incident(node, mode="OUT")
                connected_edges = [e for e in connected_edges if e != old_vessels[support]]
                valid_edges = []

                for e in connected_edges:
                    node1, node2 = es[e]
                
                    if node == node1 and abs(self.rbc_velocity[e]) > 0:
                        if self.max_particles_vessel[e] - self.flow_network.num_particles_in_vessel[e] > 0:
                            valid_edges.append(e)
                
                valid_edges = np.array(valid_edges)

                # if valid_edges.size > 0:
                #     total_flow_rate = sum(abs(self.flow_rate[e]) for e in valid_edges)
                #     probabilities = [abs(self.flow_rate[e]) / total_flow_rate for e in valid_edges]
                #     selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]
                #     new_vessels[support] = selected_edge

                if valid_edges.size > 0:
                    new_vessel = self.rbc_bifurcations(old_vessels[support], valid_edges)
                    new_vessels[support] = new_vessel
                    old_vessel = old_vessels[support]
                    self.flow_network.num_particles_in_vessel[old_vessel] -= 1
        
                    self.flow_network.num_particles_in_vessel[new_vessel] += 1

                # If the particle cannot go to the next vessel: remains stucked.
                if valid_edges.size == 0:
                    new_vessels[support] = old_vessels[support]

                new_vessels = new_vessels.astype(int)
                if new_vessels[support] < 0:
                    print(":( Negative vessel selected:", new_vessels[support])

                support += 1

            return new_vessels, index_out_particles

    def rbc_bifurcations(self, old_vessel, valid_edges):

            if valid_edges.size == 1:
                    new_vessel = valid_edges[0]

            elif valid_edges.size == 2:
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

            elif valid_edges.size > 2:
                
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
            for i, valor in enumerate(self.ht):
                if valor > 1:
                    print(f"El valor de ht en la posición {i} es mayor que uno: {valor}")
                elif valor < 0:
                    print(f"El valor de ht en la posición {i} es negativo: {valor}")

    def detect_possible_inflow_vessels(self):
        """
        Detect inflow vessels connected to inflow vertices.
        
        Returns:
        - inflow_vessels: list of vessel indices that are inflow vessels.
        """
        inflow_vessels = []

        for inflow_vertex in self.inflow_vertices:
            # Get edges (vessels) connected to the inflow vertex
            connected_edges = self.graph.incident(inflow_vertex, mode="ALL")  # Incident edges to the vertex

            for edge_index in connected_edges:
                # Check the flow direction and add the edge if it's inflow
                start, end = self.es[edge_index]  # Start and end nodes of the edge
                
                if inflow_vertex == start and abs(self.rbc_velocity[edge_index]) > 0:  # Outgoing flow
                    inflow_vessels.append(edge_index)

        return inflow_vessels

    def initialization_ghost_vessels(self):
        """
        Initializes ghost vessels and generates particles for inflow vessels.

        Returns:
        - ghost_particles: Dictionary mapping inflow vessel IDs to particle positions.
        """
        ghost_particles = {}  # To store particle positions for each inflow vessel
        k = 300 # help to define length of ghost vessels
        for vessel_id in self.inflow_vessels:
            # Get properties of the inflow vessel
            diameter = self.diameter[vessel_id]
            rbc_velocity = self.rbc_velocity[vessel_id]
            
            # Define ghost vessel properties
            ghost_length = k * abs(rbc_velocity) * self.delta_t  # Length proportional to flow
            ghost_volume = np.pi * (diameter / 2)**2 * ghost_length  # Cylindrical volume
            
            # Calculate number of particles required
            num_particles = int((ghost_volume * self.ht_boundary_condition) // self.rbc_volume)

            # Generate random particle positions in normalized coordinates [0, 1]
            positions = np.sort(np.random.rand(num_particles))

            # Store the positions in the dictionary
            ghost_particles[vessel_id] = {
                "positions": positions.tolist(),
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

        for vessel_id in active_vessels:
            # Get the ghost vessel data
            ghost_data = ghost_particles[vessel_id]
            positions = ghost_data["positions"]
            current_position = ghost_data["current_index"]  # This will track the position along the vessel
            queue_particles = ghost_data["queue"]  # Particles waiting to enter
            remaining_capacity_vessel = remaining_capacity[vessel_id]
            # Get the properties of the ghost vessel
            diameter = self.diameter[vessel_id]
            rbc_velocity = self.rbc_velocity[vessel_id]

            # Calculate the length that should be "filled" with particles in this timestep
            # The total distance the RBCs would travel in the timestep
            distance_to_fill = abs(rbc_velocity) * self.delta_t  # distance to fill in this timestep

            # Total length of the ghost vessel
            ghost_length = ghost_data["ghost_length"]

            # If the total length filled exceeds the ghost vessel length, wrap around or reset
            if (current_position + distance_to_fill) / ghost_length > 1:
                # Reset current position and recalculate particle positions
                ghost_data["current_index"] = 0  # Reset position to start
                current_position = 0
                # Redistribute the particles by recalculating the positions
                ghost_data["positions"] = np.sort(np.random.rand(ghost_data["number_particles"]))
                # Recalculate the length of the ghost vessel for the next timestep
                ghost_length = ghost_data["ghost_length"] 
                
            ghost_data["current_index"] += distance_to_fill  # Update position

            # Now, we need to determine which particles fall within the range of the distance filled
            start_position = current_position / ghost_length
            end_position = (current_position + distance_to_fill) / ghost_length

            # Get particles that are between start_position and end_position
            particles_in_range = positions[(positions >= start_position) & (positions < end_position)]
            total_particles = len(particles_in_range) + queue_particles


            if total_particles > remaining_capacity_vessel:
                # Select only the first `remaining_capacity_vessel` particles
                num_to_accept = remaining_capacity_vessel
                accepted_particles = np.sort(particles_in_range)[:num_to_accept]
                excess_particles = total_particles - remaining_capacity_vessel

                # Update the queue with the excess particles
                ghost_data["queue"] = excess_particles

                # Place the excess particles at the max position for the next timestep
                new_positions = np.concatenate([
                    ghost_data["positions"],
                    np.full(excess_particles, end_position)
                ])
                ghost_data["positions"] = np.sort(new_positions)

            else:
                # If all particles can fit, reset the queue and accept all particles
                accepted_particles = particles_in_range
                ghost_data["queue"] = 0

            # Append the accepted particles to inflow_particles
            for pos in accepted_particles:
                local_position = pos - start_position  # Calculate local position
                inflow_particles.append([vessel_id, local_position])

            self.flow_network.num_particles_in_vessel[vessel_id] += len(accepted_particles)

            timestep_particles_count += len(accepted_particles)

        return inflow_particles, timestep_particles_count
