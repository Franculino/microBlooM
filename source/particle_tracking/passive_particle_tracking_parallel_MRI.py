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
import pickle

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
        self.volume = self.get_volumes()
        
        self.use_tortuosity = PARAMETERS["use_tortuosity"]
        self.parallel = PARAMETERS['parallel']
        self.N_particles = self._PARAMETERS["initial_number_particles"]
        self.initial_particle_tube = self._PARAMETERS["initial_vessels"]
        self.times_basic_delta_t = self._PARAMETERS['times_basic_delta_t']
        self.delta_t = self.times_basic_delta_t * abs(self.length).min()/(abs(self.rbc_velocity).max())
        self.N_timesteps =  self._PARAMETERS["N_timesteps"]
        self.inflow_vertices, self.outflow_vertices = self.detect_inflow_outflow_vertices()
        self.inflow_vertices = np.array(self.inflow_vertices)
        self.out_particles = []
        self.particles_frequency = PARAMETERS["particles_frequency"]
        self.pkl_path = self._PARAMETERS["pkl_path_igraph"]

        self.indices_rbc_negativa = np.where(self.rbc_velocity < 0)[0]
        self.es[self.indices_rbc_negativa] = self.es[self.indices_rbc_negativa][:, ::-1]

        self.initial_particles_coords = np.zeros((self.N_particles, 3))
        self.initial_local_coord = np.full(self.N_particles, 0.5) 
        self.initialization_constant = PARAMETERS["initialization_constant"]
        self.rbc_volume = PARAMETERS["rbc_volume"]

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
        self.initialize_particles_evolution_constant()

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
    def get_intervals(self):
        """
        Generate interval (time between particles entering through each inflow vertex) based on the user's selection.
        If the user wants to specify them, they are read from PARAMETERS.
        Otherwise, they are generated based on the flow_rate.
        """
        interval_mode = self._PARAMETERS.get("interval_mode", 2)  # 0 = manual, 1 = automatic

        if interval_mode == 0:
            # If the user specifies the intervals
            intervals = list(np.full(len(self.inflow_vertices), self.particles_frequency))
            return intervals
        elif interval_mode == 1:
            # Generate intervals based on the flow_rate of inflow vertices
            flow_rate_inflow = []

            # Use self.inflow_vertices directly, which is already calculated
            for vertex in self.inflow_vertices:
                edges_connected = self.graph.incident(vertex, mode='in')
                total_flow_rate = np.sum(self.flow_rate[edges_connected])
                flow_rate_inflow.append(total_flow_rate)

            flow_rate_inflow = np.array(flow_rate_inflow)
            intervals_normalized = abs(np.max(abs(flow_rate_inflow)) / flow_rate_inflow)
            intervals_normalized = (intervals_normalized / np.min(intervals_normalized))
            # avoid rollover of numpy integers
            intervals = np.clip(intervals_normalized, a_min=None, a_max=1e6)
            intervals = list(intervals.astype(int))
            

            return intervals
        else:
            # Default case if no valid mode is selected
            raise ValueError(f"Invalid interval_mode: {interval_mode}")

    def initialize_particles_evolution_constant(self):
        total_particles_added = self.calculate_total_particles_added()
        self.initial_particles_per_vessel = np.zeros(len(self.es), dtype = int)
        
        for vessel_id in range(len(self.es)):
            self.initial_particles_per_vessel[vessel_id] = int((self.volume[vessel_id] * self.initialization_constant)  // self.rbc_volume)
        
        self.N_particles = sum(self.initial_particles_per_vessel)
        # Total number of particles
        self.N_particles_total = int(self.N_particles + total_particles_added)
        print('Total number of simulated particles:', self.N_particles_total)
        self.particles_per_timestep = self.predict_particles(self.N_particles, self.N_timesteps + 1, self.get_intervals())

        # Adjust particles_per_timestep
        self.particles_per_timestep = np.insert(self.particles_per_timestep, 0, self.N_particles)
        # No particles inflowing in timestep 1
        num_ones = self.get_intervals().count(1)
        self.particles_per_timestep[1:] -= num_ones
        self.particles_per_timestep[self.particles_per_timestep < 0] = 0

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
      

    def initialize_particles_evolution(self):
        """
        Initialize the particles' evolution over time using the defined time steps, intervals, and conditions.
        """
        total_particles_added = self.calculate_total_particles_added()

        # Total number of particles
        self.N_particles_total = int(self.N_particles + total_particles_added)
        print('Total number of simulated particles:', self.N_particles_total)
        self.particles_per_timestep = self.predict_particles(self.N_particles, self.N_timesteps + 1, self.get_intervals())

        # Adjust particles_per_timestep
        self.particles_per_timestep = np.insert(self.particles_per_timestep, 0, self.N_particles)

        # No particles inflowing in timestep 1
        num_ones = self.get_intervals().count(1)
        self.particles_per_timestep[1:] -= num_ones
        self.particles_per_timestep[self.particles_per_timestep < 0] = 0

        # Initialize particles evolution matrix
        self.particles_evolution = np.zeros((self.N_particles_total, self.N_timesteps + 1, 2), dtype=object)
        self.particles_evolution[:self.N_particles, 0, :] = self.initial_particles()
        self.particles_evolution[self.N_particles:, :, :] = np.nan
        self.inactive_particles = np.zeros(self.N_particles_total, dtype=bool)

    def predict_particles(self, N_particles, N_timesteps, intervals):
        """
        Predict the number of particles entering the network at each timestep based on intervals.
        """
        def count_multiples(N, intervals):
            counts = np.zeros(N, dtype=int)
            for interval in intervals:
                indices = np.arange(interval - 1, N, interval)
                counts[indices] += 1
            return counts

        # Accumulate the number of particles added over time
        counts = count_multiples(N_timesteps, intervals)
        added_particles = np.array(list(itertools.accumulate(counts)))
        particles_per_timestep = added_particles + N_particles

        return particles_per_timestep

    def calculate_total_particles_added(self):
        """
        Calculate the total number of particles added over time based on intervals.
        """
        num_ones = self.get_intervals().count(1)
        total_particles_added = np.sum(np.floor_divide(self.N_timesteps + 1, self.get_intervals())) - num_ones
        return total_particles_added

    def initial_particles(self):
        """
        Initialize the particles position of those particles that start inside the network at time 0.
        """
        initial_particle_tube = np.array(self._PARAMETERS["initial_vessels"]).astype(int)
        initial_local_coord = np.full(self.N_particles, 0.5)
        initial = np.array([[int(tube), coord] for tube, coord in zip(initial_particle_tube, initial_local_coord)])
        return initial
    
    def evolve_particles(self):
        """Evolve particles across each timestep. Computes the movement of every particles in the net"""
        
        for t in range(1, self.N_timesteps + 1):
            if t ==1:
                self.delta_t = 0.0005
                print(f'Timestep 1500: delta_t updated to {self.delta_t}')
            # print('Timestep: ', self.delta_t)
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
            new_velocities = abs(self.rbc_velocity[new_vessels])

            second_prediction_position = new_velocities * remaining_time_positive / self.length[new_vessels]
            self.particles_evolution[change_vessel_positive, t, 1] = second_prediction_position
            if np.any(second_prediction_position > 1):
                print("A particles is not being propagated correctly: you should decrease the timestep")

            # Manage particles that exit the network.
            if index_out_particles:
                for idx in index_out_particles:
                    self.out_particles.append(change_vessel_positive[idx])
                    self.particles_evolution[change_vessel_positive[idx], t - 1:, :] = np.nan
                    self.inactive_particles[change_vessel_positive[idx]] = True

            # Initialize particles entering the network in the next timestep
            if self.particles_per_timestep[t + 1] > self.particles_per_timestep[t]:
                real_timestep = t + 1
                exact_divisors = [i for i in range(len(self.intervals)) if real_timestep % self.intervals[i] == 0]
                nodes_inflowing = self.inflow_vertices[exact_divisors]
                vessels_inflowing = self.select_vessels_inflow(nodes_inflowing, self.graph, self.flow_network.edge_list)
                number_inflowing_particles = self.particles_per_timestep[t + 1] - self.particles_per_timestep[t]
                self.particles_evolution[self.particles_per_timestep[t]:self.particles_per_timestep[t + 1], t, 0] = vessels_inflowing.astype(int)
                self.particles_evolution[self.particles_per_timestep[t]:self.particles_per_timestep[t + 1], t, 1] = 0.0
        # self.save_particles_evolution_to_excel()

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
                support += 1
                continue  
            connected_edges = graph.incident(node, mode="OUT")
            connected_edges = [e for e in connected_edges if e != old_vessels[support]]
            valid_edges = []

            for e in connected_edges:
                node1, node2 = es[e]
            
                if node == node1 and abs(self.rbc_velocity[e]) > 0:
                    valid_edges.append(e)
            
            valid_edges = np.array(valid_edges)

            if valid_edges.size > 0:
                total_flow_rate = sum(abs(self.flow_rate[e]) for e in valid_edges)
                probabilities = [abs(self.flow_rate[e]) / total_flow_rate for e in valid_edges]
                selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]
                new_vessels[support] = selected_edge
            
            support += 1
        
        return new_vessels, index_out_particles

    def select_vessels_inflow(self, vertices, graph, es):
        """
        Selects the vessels into which the particles inflow based on the connected vessels from the inflow vertices.
        
        Parameters:
        - vertices: list of inflow vertices where particles will enter.
        - graph: the network graph representing the flow.
        - es: edge list representing the vessels in the network.
        
        Returns:
        - new_vessels: array of selected vessels into which the particles will inflow.
        """
        new_vessels = np.zeros(len(vertices))
        support = 0  # To track the index for the new_vessels array

        for node in vertices:
            # Get the edges (vessels) connected to the node with outgoing flow
            connected_edges = graph.incident(node, mode="OUT")
            valid_edges = []

            # Check each connected edge and validate its flow direction
            for e in connected_edges:
                node1, node2 = es[e]
                # Ensure that the flow is outgoing from the current node
                if node == node1 and abs(self.rbc_velocity[e]) > 0:
                    valid_edges.append(e)
            
            # Convert the list to a numpy array for easier processing
            valid_edges = np.array(valid_edges)

            if valid_edges.size > 0:
                # If there's more than one valid edge, select based on the flow rate probabilities
                total_flow_rate = sum(abs(self.flow_rate[e]) for e in valid_edges)
                probabilities = [abs(self.flow_rate[e]) / total_flow_rate for e in valid_edges]

                # Randomly select an edge (vessel) based on the calculated probabilities
                selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]
                new_vessels[support] = selected_edge
            else:
                # Handle case where no valid edge is found (e.g., particle stuck at an inflow node)
                print(f"No valid vessel found for node {node}. Particle might be stuck.")
                new_vessels[support] = -1  # Optional: use -1 or another flag to indicate no valid vessel
            
            support += 1
        
        return new_vessels

    def create_vtk_particles_per_timestep(self, output_dir):
        num_particles, num_timesteps, _ = self.particles_evolution_global.shape
        
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
            valid_mask = ~np.isnan(self.particles_evolution_global[:, t, 0])
            valid_positions = self.particles_evolution_global[valid_mask, t, :]
            
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

            self.particles_evolution_global_local = np.full((local_particles_count, self.N_timesteps + 1, 3), np.nan)
            
            if self.use_tortuosity == 1:
                if rank == 0:

                    for vessel_id in range(len(self.es)):
                        vessel_points = np.array(self.points[vessel_id])
                        vessel_lengths = np.array(self.lengths[vessel_id])
                        vessel_total_length = self.length[vessel_id]

                        if vessel_id in self.indices_rbc_negativa:
                            vessel_points = vessel_points[::-1]
                            vessel_lengths = vessel_lengths[::-1]

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

                    self.particles_evolution_global_local[local_idx, t] = particle_global_position

            self.particles_evolution_global = None
            if rank == 0:
                self.particles_evolution_global = np.zeros((self.N_particles_total, self.N_timesteps + 1, 3))

            sendcounts = np.array([particles_per_process] * size)
            sendcounts[-1] = self.N_particles_total - (size - 1) * particles_per_process
            displacements = np.array([i * particles_per_process for i in range(size)])

            comm.Gatherv(
                self.particles_evolution_global_local, 
                [self.particles_evolution_global, 
                sendcounts * (self.N_timesteps + 1) * 3, 
                displacements * (self.N_timesteps + 1) * 3, 
                MPI.DOUBLE], 
                root=0
            )
            if rank == 0:
                return self.particles_evolution_global
            else:
                return None  
        else:
            
            # SEQUENTIAL IMPLEMENTATION 

            self.particles_evolution_global = np.full((self.N_particles_total, self.N_timesteps + 1, 3), np.nan)
            self.vessel_data = {}

            if self.use_tortuosity == 1:
                for vessel_id in range(len(self.es)):
                    vessel_points = np.array(self.points[vessel_id])
                    vessel_lengths = np.array(self.lengths[vessel_id])
                    vessel_total_length = self.length[vessel_id]

                    if vessel_id in self.indices_rbc_negativa:
                        vessel_points = vessel_points[::-1]
                        vessel_lengths = vessel_lengths[::-1]

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

                        self.particles_evolution_global[p, t] = particle_global_position
                return self.particles_evolution_global

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

                        self.particles_evolution_global[p, t] = particle_global_position
                return self.particles_evolution_global

            else:
                raise ValueError(f"Invalid use_tortuosity: {self.use_tortuosity}. It must be either 0 or 1.")


    def compute_velocity_components(self):
        """
        This function generates three matrices (one for each velocity component: x, y, z)
        from `particles_evolution`. Each matrix has particles as rows and timesteps as columns.
        
        Returns:
            - velocity_x: Matrix with the x component of velocity for each particle at each timestep.
            - velocity_y: Matrix with the y component of velocity for each particle at each timestep.
            - velocity_z: Matrix with the z component of velocity for each particle at each timestep.
        """
        start_time = 0
        end_time = 400
        n_timesteps_to_keep = end_time - start_time + 1
        # Initialize the velocity matrices for x, y, z with NaN
        self.velocity_x = np.full((self.N_particles_total, n_timesteps_to_keep), np.nan)
        self.velocity_y = np.full((self.N_particles_total, n_timesteps_to_keep), np.nan)
        self.velocity_z = np.full((self.N_particles_total, n_timesteps_to_keep), np.nan)

        for vessel_id in range(len(self.es)):
            vessel_points = np.array(self.points[vessel_id])
            vessel_lengths = np.array(self.lengths[vessel_id])
            vessel_total_length = self.length[vessel_id]

            if vessel_id in self.indices_rbc_negativa:
                vessel_points = vessel_points[::-1]
                vessel_lengths = vessel_lengths[::-1]

            normalized_lengths = np.cumsum(vessel_lengths) / vessel_total_length
            normalized_lengths = np.insert(normalized_lengths, 0, 0)

            self.vessel_data[vessel_id] = {
                'points': vessel_points,
                'normalized_lengths': normalized_lengths
            }

        # List to store the normalized direction vectors of each vessel, multiplied by the RBC velocity
        for p in range(self.N_particles_total):
            if p % 100 == 0:
                print(f"Processing particle {p} of {self.N_particles_total}...")
            
            for t in range(n_timesteps_to_keep):
                actual_timestep = start_time + t 
                vessel_id = self.particles_evolution[p, actual_timestep, 0]
                local_coord = self.particles_evolution[p, actual_timestep, 1]

                # Verificar si la partícula está activa (si no es NaN)
                if np.isnan(vessel_id) or np.isnan(local_coord):
                    continue

                vessel_id = int(vessel_id)

                # Obtener los puntos del vaso y las longitudes normalizadas
                vessel_points = self.vessel_data[vessel_id]['points']
                normalized_lengths = self.vessel_data[vessel_id]['normalized_lengths']

                # Encontrar los puntos adyacentes más cercanos basados en la coordenada local
                point_idx = np.searchsorted(normalized_lengths, local_coord, side='right') - 1
                point_idx = min(point_idx, len(vessel_points) - 2)  # Asegurarse de no exceder el índice

                point_start = vessel_points[point_idx]
                point_end = vessel_points[point_idx + 1]

                # Coordenadas normalizadas de los puntos
                local_start = normalized_lengths[point_idx]
                local_end = normalized_lengths[point_idx + 1]

                # Calcular el vector director del tramo del vaso (punto_end - punto_start)
                direction_vector = point_end - point_start
                norm = np.linalg.norm(direction_vector)

                # Normalizar el vector director si su norma no es cero
                if norm != 0:
                    direction_vector_normalized = direction_vector / norm
                else:
                    direction_vector_normalized = np.zeros_like(direction_vector)

                # Multiplicar el vector normalizado por la velocidad del RBC en ese vaso
                rbc_velocity = abs(self.rbc_velocity[vessel_id])
                velocity_vector = direction_vector_normalized * rbc_velocity

                # Guardar las componentes de velocidad en las matrices correspondientes
                self.velocity_x[p, t] = velocity_vector[0]
                self.velocity_y[p, t] = velocity_vector[1]
                self.velocity_z[p, t] = velocity_vector[2]
        
        print("Computation of vlocities is complete.")
        return self.velocity_x, self.velocity_y, self.velocity_z


    def compute_nkind_matrix(self):
        """
        This function generates a matrix where each row corresponds to a particle, and each column corresponds
        to a timestep. The matrix stores the 'nkind' value of the vessel in which each particle is located at each timestep.
    
        Returns:
            - nkind_matrix: A matrix where each entry stores the 'nkind' value for the vessel in which the particle is located.
        """
        # Load the .pkl file that contains the 'nkind' attribute for each edge
        with open(self.pkl_path, 'rb') as file:
            graph_data = pickle.load(file)

        # Extract the 'nkind' attribute from the edges
        self.nkind = graph_data.es['nkind']  # Assuming 'nkind' is stored as an attribute for each edge

        start_time = 0
        end_time = 400
        n_timesteps_to_keep = end_time - start_time + 1

        # Initialize the matrix with NaN values
        self.nkind_matrix = np.full((self.N_particles_total, n_timesteps_to_keep), -1, dtype=np.int32)

        # Fill the matrix based on the particles' positions in 'particles_evolution'
        for p in range(self.N_particles_total):

            if p % 100 == 0:
                print(f"Processing particle {p} of {self.N_particles_total}...")

            for t in range(n_timesteps_to_keep): 
                actual_timestep = start_time + t 
                # Identify the vessel where the particle is located at the current timestep
                vessel_id = self.particles_evolution[p, actual_timestep, 0]

                # Check if the particle is active (i.e., not NaN)
                if np.isnan(vessel_id):
                    continue

                # Convert vessel_id to integer
                vessel_id = int(vessel_id)

                # Get the 'nkind' value of the corresponding vessel
                nkind_value = self.nkind[vessel_id]

                # Fill the nkind matrix
                self.nkind_matrix[p, t] = int(nkind_value)
        print("Computation of nkind_matrix is complete.")

        return self.nkind_matrix

    def save_global_coordinates_to_csv(self):
        """
        Save the global coordinates matrices (x, y, z) to CSV files with proper number formatting and semicolon separator.
        """
        # Guardar las matrices de coordenadas globales con formato apropiado y separador ';'
        pd.DataFrame(self.particles_evolution_global[:, :, 0]).to_csv('data/network/global_x.csv', index=False, header=False, sep=';', float_format='%.10f')
        pd.DataFrame(self.particles_evolution_global[:, :, 1]).to_csv('data/network/global_y.csv', index=False, header=False, sep=';', float_format='%.10f')
        pd.DataFrame(self.particles_evolution_global[:, :, 2]).to_csv('data/network/global_z.csv', index=False, header=False, sep=';', float_format='%.10f')

        print("Global coordinates saved to CSV files with proper formatting.")

    def save_matrices_to_csv(self):
        """
        Save the velocity and nkind matrices to CSV files with proper number formatting and semicolon separator.
        """
        # Guardar las matrices de velocidad
        pd.DataFrame(self.velocity_x).to_csv('data/network/velocity_x.csv', index=False, header=False, sep=';', float_format='%.10f')
        pd.DataFrame(self.velocity_y).to_csv('data/network/velocity_y.csv', index=False, header=False, sep=';', float_format='%.10f')
        pd.DataFrame(self.velocity_z).to_csv('data/network/velocity_z.csv', index=False, header=False, sep=';', float_format='%.10f')
        
        # Guardar la matriz nkind
        pd.DataFrame(self.nkind_matrix).to_csv('data/network/nkind_matrix.csv', index=False, header=False, sep=';', float_format='%.0f')

        print("Matrices saved to CSV files with proper formatting.")

    def save_particles_evolution_to_excel(self):
        # Extraer las dimensiones del array
        N_particles_total, N_timesteps_plus_1, _ = self.particles_evolution.shape

        # Crear una lista de columnas: una columna por cada timestep
        columns = [f'Timestep_{t}' for t in range(N_timesteps_plus_1)]

        # Inicializar una lista para almacenar los datos de cada partícula
        data = []

        # Recorrer cada partícula y combinar (vessel, position) en una misma celda para cada timestep
        for i in range(N_particles_total):
            particle_data = []
            for t in range(N_timesteps_plus_1):
                vessel = self.particles_evolution[i, t, 0]  # valor del vaso sanguíneo
                position = self.particles_evolution[i, t, 1]  # valor de la posición local
                # Concatenar en un formato (vessel, position)
                particle_data.append(f'({vessel}, {position})')
            data.append(particle_data)

        # Crear un DataFrame a partir de los datos
        df = pd.DataFrame(data, columns=columns)

        # Guardar el DataFrame en un archivo Excel
        file_name = "data/network/particles_evolution_local.xlsx"
        df.to_excel(file_name, index=False)

        print(f"El archivo '{file_name}' se ha guardado correctamente.")

