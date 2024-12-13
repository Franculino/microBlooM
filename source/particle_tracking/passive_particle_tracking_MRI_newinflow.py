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
        self.bulk_velocity = self.flow_rate / (np.square(self.diameter) * np.pi / 4)
        self.pressure = flow_network.pressure
        self.volume = self.get_volumes()
        self.flow_network.volume = self.volume
        # self.max_particles_vessel = np.floor(self.volume / self.flow_network.rbc_volume).astype(int)
        
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

        self.outflow_vessels = self.detect_possible_outflow_vessels()
        self.total_exited_volume = 0
        self.exiting_volume_per_vessel = self.delta_t * abs(self.flow_rate[self.outflow_vessels])
        self.exiting_volume_per_timestep = np.sum(self.exiting_volume_per_vessel)
        total_volume_network = np.sum(self.volume)
        self.timesteps_until_steadystate = 1.4 * (total_volume_network // self.exiting_volume_per_timestep) + 1
        print("timesteps until steady state:", self.timesteps_until_steadystate)
        self.N_timesteps =  int(self.timesteps_until_steadystate)
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
        self.outflow_vessels = self.detect_possible_outflow_vessels()
        self.ghost_particles = self.initialization_ghost_vessels()
        self.total_added_particles = 0


    def initialize_particles_with_hematocrit(self):

            self.initial_particles_per_vessel = np.zeros(len(self.es), dtype = int)
            for vessel_id in range(len(self.es)):
                self.initial_particles_per_vessel[vessel_id] = int((self.volume[vessel_id] * self.ht_initial)  // self.rbc_volume)
            
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
            self.N_particles_total += 5000
            
            # Expansión de particles_evolution
            new_particles_evolution = np.zeros((self.N_particles_total, self.N_timesteps + 1, 2), dtype=object)
            new_particles_evolution[:self.particles_evolution.shape[0], :, :] = self.particles_evolution 
            new_particles_evolution[self.particles_evolution.shape[0]:, :, :] = np.nan
            
            # Expansión de inactive_particles
            new_inactive_particles = np.zeros(self.N_particles_total, dtype=bool)
            new_inactive_particles[:self.inactive_particles.shape[0]] = self.inactive_particles
            
    

            # Asignar los nuevos arrays expandidos
            self.particles_evolution = new_particles_evolution
            self.inactive_particles = new_inactive_particles
            
            print("The arrays have been updated: ", self.N_particles_count, self.N_particles_total)

    def evolve_particles(self):
        """Evolve particles across each timestep. Computes the movement of every particles in the net"""
        # collision_count = 0
        # bifurcation_count = 0
        # self.particle_size = np.zeros(self.particles_evolution.shape[0])
        # print('Timestep: ', self.delta_t)

        for t in range(1, self.N_timesteps + 1):
            self.delta_t = self.times_basic_delta_t * abs(self.length).min()/(abs(self.rbc_velocity).max())
            print('Delta_t = ', self.delta_t)

            # for vessel_idx in range(len(self.flow_network.edge_list)):
            #     # Actualizar el hematocrito, número de partículas y volumen para cada vaso en este timestep
            #     self.hematocrit_evolution[vessel_idx, t-1] = self.flow_network.ht[vessel_idx]
            #     self.num_particles_evolution[vessel_idx, t-1] = self.flow_network.num_particles_in_vessel[vessel_idx]
            #     self.volume_evolution[vessel_idx, t-1] = self.volume[vessel_idx]
                
            # Determine active particles for this timestep
            current_timestep_particles = self.particles_evolution[:self.N_particles_count, t-1, 0].astype(float)
            active_particles = np.where(~np.isnan(current_timestep_particles))[0]

            # Calculate the total distance to travel for all particles
            initial_vessels_per_iteration = self.particles_evolution[active_particles, t - 1, 0].astype(int)
            particle_diameters = self.diameter[initial_vessels_per_iteration]
            # self.particle_size[active_particles] = self.rbc_volume / (np.pi * particle_diameters**2 / 4)
            velocities_per_iteration = abs(self.rbc_velocity[initial_vessels_per_iteration])
            length_per_iteration = self.length[initial_vessels_per_iteration]
            distance_to_travel = velocities_per_iteration * self.delta_t

            # Calculate the remaining distance in the current vessels
            local_position_per_iteration = self.particles_evolution[active_particles, t - 1, 1]
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

            new_velocities = abs(self.rbc_velocity[new_vessels[moving_particles_idx]])

            second_prediction_position = new_velocities * remaining_time_positive[moving_particles_idx] / self.length[new_vessels[moving_particles_idx]]
            self.particles_evolution[moving_particles, t, 1] = second_prediction_position

            new_vessel_diameters = self.diameter[new_vessels[moving_particles_idx]]
            # self.particle_size[moving_particles] = self.rbc_volume / (np.pi * new_vessel_diameters**2 / 4)

            if np.any(second_prediction_position > 1):
                print("A particles is not being propagated correctly: you should decrease the timestep")

            
            # Initialize particles entering the network in the next timestep
        
            # vessels_inflowing = self.select_vessels_inflow(self.inflow_vertices, self.graph, self.flow_network.edge_list).astype(int)
            # remaining_capacity = self.max_particles_vessel[self.inflow_vessels] - self.flow_network.num_particles_in_vessel[self.inflow_vessels]
            
            inflow_particles, number_inflowing_particles = self.update_ghost_particles(self.ghost_particles, self.inflow_vessels)
            self.N_particles_count = int(self.N_particles_count + number_inflowing_particles)
            self.expand_arrays_if_needed()

            if number_inflowing_particles > 0:
                vessels = inflow_particles[:,0]
                local_positions = inflow_particles[:,1]
                
                self.particles_evolution[int(self.N_particles_count - number_inflowing_particles): self.N_particles_count, t, 0] = vessels.astype(int)
                self.particles_evolution[int(self.N_particles_count - number_inflowing_particles): self.N_particles_count, t, 1] = local_positions
            print('Timesetp: ', t, '  Number of particles: ', self.flow_network.num_particles_in_vessel[0] )
        print('Ya acabo el bucle')
        print('Number of particles added:', self.total_added_particles)
        # print('Bifurcations:', bifurcation_count)
        # print('Collisions:', collision_count)
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

    def detect_possible_outflow_vessels(self):
        """
        Detect outflow vessels connected to outflow vertices.
        
        Returns:
        - outflow_vessels: list of vessel indices that are outflow vessels.
        """
        outflow_vessels = []

        for outflow_vertex in self.outflow_vertices:
            # Get edges (vessels) connected to the inflow vertex
            connected_edges = self.graph.incident(outflow_vertex, mode="ALL")  # Incident edges to the vertex

            for edge_index in connected_edges:
                # Check the flow direction and add the edge if it's inflow
                start, end = self.es[edge_index]  # Start and end nodes of the edge
                
                if outflow_vertex == end and abs(self.rbc_velocity[edge_index]) > 0:  # Outgoing flow
                    outflow_vessels.append(edge_index)

        return outflow_vessels
  
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
            bulk_velocity = self.bulk_velocity[vessel_id]
            
            # Define ghost vessel properties
            ghost_length = k * abs(bulk_velocity) * self.delta_t  # Length proportional to flow
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

    def update_ghost_particles(self, ghost_particles, active_vessels):
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
            # Get the properties of the ghost vessel
            diameter = self.diameter[vessel_id]
            bulk_velocity = self.bulk_velocity[vessel_id]

            # Calculate the length that should be "filled" with particles in this timestep
            # The total distance the RBCs would travel in the timestep
            distance_to_fill = abs(bulk_velocity) * self.delta_t  # distance to fill in this timestep

            # Total length of the ghost vessel
            ghost_length = ghost_data["ghost_length"]

            # If the total length filled exceeds the ghost vessel length, wrap around or reset
            if (current_position + distance_to_fill) / ghost_length > 1:
                # Reset the current position since the particles have traversed the entire ghost vessel length
                ghost_data["current_index"] = 0  # Start position at the beginning
                current_position = 0  # Reset current position for the next timestep
                # Recalculate ghost vessel properties based on updated RBC velocity
                new_ghost_length = k * abs(bulk_velocity) * self.delta_t  # Compute the new length of the ghost vessel
                new_ghost_volume = np.pi * (diameter / 2)**2 * new_ghost_length  # Calculate the volume of the ghost vessel cylinder
                new_num_particles = int((new_ghost_volume * self.ht_boundary_condition) // self.rbc_volume)  # Determine the required number of particles

                # Update the ghost vessel's properties in the dictionary
                ghost_data["ghost_length"] = new_ghost_length  # Assign the new length
                ghost_data["number_particles"] = new_num_particles  # Assign the recalculated number of particles
                ghost_data["ghost_volume"] = new_ghost_volume  # Assign the recalculated number of particles
                ghost_data["positions"] = np.sort(np.random.rand(new_num_particles))  # Generate new random particle positions
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
            # Place the excess particles at the max position for the next timestep

            # Append the accepted particles to inflow_particles
            for pos in particles_in_range:
                local_position = (-pos + end_position) * ghost_length / self.length[vessel_id]  # Calculate local position
                inflow_particles.append([vessel_id, local_position])

            self.flow_network.num_particles_in_vessel[vessel_id] += len(particles_in_range)

            timestep_particles_count += len(particles_in_range)
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

    def save_steady_state(self):
        # Obtén el último timestep del steady state
        last_timestep = self.N_timesteps
        active_particles = np.where(~self.inactive_particles)[0]
        
        # Guarda las posiciones y los estados de las partículas activas
        self.steady_state_particles = {
            "active_particles": active_particles,
            "positions": self.particles_evolution[active_particles, last_timestep, :].copy()
        }

    def initialize_from_steady_state(self):
        """
        Reinitialize matrix partcle_evolurion using the previously computed steady state.
        """
        self.N_timesteps =  self._PARAMETERS["N_timesteps"]
        self.N_particles = len(self.steady_state_particles["positions"])
        
        self.delta_t = 0.0005

        # Total number of particles
        self.N_particles_total = int(self.N_particles + 3000)
    

        # Adjust particles_per_timestep
        

        # No particles inflowing in timestep 1
        # Reinicia la matriz de evolución
        self.particles_evolution = np.zeros((self.N_particles_total, self.N_timesteps + 1, 2), dtype=object)
        self.particles_evolution[:, :, :] = np.nan  # Partículas inactivas están llenas de NaN
        
        # Coloca las partículas activas en el nuevo timestep inicial
        active_particles = self.steady_state_particles["active_particles"]
        positions = self.steady_state_particles["positions"]
        
        self.particles_evolution[:self.N_particles, 0, :] = positions
        self.inactive_particles = np.ones(self.N_particles_total, dtype=bool)
        self.inactive_particles[:self.N_particles] = False

    def save_particles_evolution_to_csv(self):
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

        # Guardar el DataFrame en un archivo CSV
        file_name = "data/network/particles_evolution_local.csv"
        df.to_csv(file_name, index=False)

        print(f"El archivo '{file_name}' se ha guardado correctamente.")