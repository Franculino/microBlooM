import numpy as np
import random
import igraph as ig
from types import MappingProxyType
from source.flow_network import FlowNetwork
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib.pyplot as plt


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

        
        self.graph = ig.Graph()  # Initialize the graph
        self.graph.add_vertices(self.vs_coords.shape[0])  # Add vertices
        self.graph.add_edges(self.es)  # Add edges

        # Asignar atributos a los vértices y aristas
        self.graph.vs['xyz'] = self.vs_coords.tolist()  # Coordinates of the vertices
        self.graph.es['length'] = self.length  # Length of the edges
        self.graph.es['diameter'] = self.diameter  # Diameter of the edges
        self.graph.es['flow_rate'] = self.flow_rate  # Flow rate through the edges
        self.graph.es['rbc_velocity'] = self.rbc_velocity # Rbc_velocity

        self.N_particles = self._PARAMETERS["initial_number_particles"]
        self.initial_particle_tube = self._PARAMETERS["initial_vessels"]
        self.delta_t = self.length.min()/(self.rbc_velocity.max())
        self.N_timesteps =  self._PARAMETERS["N_timesteps"]
        self.inflow_vertices =  self._PARAMETERS["inflow_vertices"]
        self.outflow_vertices =  self._PARAMETERS["outflow_vertices"]

        self.initial_particles_coords = np.zeros((self.N_particles, 3))
        self.initial_local_coord = np.full(self.N_particles, 0.5)  # Referencia del tubo adimensional
        self.particles_evolution = np.zeros((self.N_particles, self.N_timesteps + 3), dtype=object)
        self.particles_evolution[:, 0] = np.arange(self.N_particles)
        self.particles_evolution[:, 1] = self.initial_particle_tube[:]

        self.initialize_particles()

    def initialize_particles(self):
        """Initializes the particle positions."""
        for i in range(self.N_particles):
            y, x = self._calculate_edge_direction(self.initial_particle_tube[i])
            delta_x, delta_y = self._calculate_particle_displacement(i, x, y)
            self.initial_particles_coords[i] = self.vs_coords[self.es[self.initial_particle_tube[i], 0]] + [delta_x, delta_y, 0]
            self.particles_evolution[i, 2] = self.initial_particles_coords[i]

    def _calculate_particle_displacement(self, particle_index, x, y):
        """Calculate displacement of a particle within an edge."""
        delta_x = self.initial_local_coord[particle_index] * self.length[self.initial_particle_tube[particle_index]] * np.cos(np.arctan2(y, x))
        delta_y = self.initial_local_coord[particle_index] * self.length[self.initial_particle_tube[particle_index]] * np.sin(np.arctan2(y, x))
        return delta_x, delta_y

    def _calculate_edge_direction(self, tube_index):
        """Calculate the x and y components of the direction of a given edge."""
        y = self.vs_coords[self.es[tube_index, 1], 1] - self.vs_coords[self.es[tube_index, 0], 1]
        x = self.vs_coords[self.es[tube_index, 1], 0] - self.vs_coords[self.es[tube_index, 0], 0]
        return y, x

    def simulate(self):
        """Run the simulation over the defined number of timesteps."""
        for t in range(1, self.N_timesteps + 1):
            self._generate_particles_at_inflows(t)
            self._update_particles(t)

    def _generate_particles_at_inflows(self, timestep):
        """Generate new particles at inflow vertices."""
        for inflow_vertex, interval in self.inflow_vertices.items():
            if timestep % interval == 0:
                valid_edges = self._find_valid_edges(inflow_vertex, -1)
                if len(valid_edges) > 0:
                    selected_edge = self._select_next_edge(valid_edges)
                    new_position = self._generate_particle_position(selected_edge)
                    self._add_new_particle(selected_edge, new_position, timestep)

    def _find_valid_edges(self, current_node, current_tube):
        """Find valid edges for a particle to move to from a given node."""
        connected_edges = self.graph.incident(current_node, mode="OUT")
        connected_edges = [e for e in connected_edges if e != current_tube]
        valid_edges = []

        for e in connected_edges:
            node1, node2 = self.es[e]

            if current_node == node1 and self.rbc_velocity[e] > 0:
                valid_edges.append(e)
            elif current_node == node2 and self.rbc_velocity[e] < 0:
                valid_edges.append(e)
        return np.array(valid_edges)

    def _select_next_edge(self, valid_edges):
        """Select the next edge for a particle based on a probability associated to flow rates."""
        total_flow_rate = sum(abs(self.flow_rate[e]) for e in valid_edges)
        probabilities = [abs(self.flow_rate[e]) / total_flow_rate for e in valid_edges]
        selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]
        return selected_edge
    
    def _generate_particle_position(self, selected_edge):
        """Generate the initial position for a new particle."""
        y, x = self._calculate_edge_direction(selected_edge)
        new_position = self.vs_coords[self.es[selected_edge, 0]] + np.array([x, y, 0]) * 1e-09
        return new_position

    def _add_new_particle(self, selected_edge, new_position, timestep):
        """Add a new particle to the particles evolution matrix."""
        new_particle_index = len(self.particles_evolution)
        new_row = np.full((1, self.N_timesteps + 3), np.nan, dtype=object)
        new_row[0, 0] = new_particle_index
        new_row[0, 1] = selected_edge
        new_row[0, 2 + timestep - 1] = new_position
        new_row[0, 2 + timestep:] = 0
        self.particles_evolution = np.vstack([self.particles_evolution, new_row])

    def _update_particles(self, timestep):
        """Update the positions of all particles for a given timestep."""
        for i in range(len(self.particles_evolution)):
            current_position = self.particles_evolution[i, timestep + 2 - 1]
            current_tube = int(self.particles_evolution[i, 1])
            previous_tube = current_tube

            if isinstance(current_position, np.ndarray):
                print(current_position, current_tube,  self.delta_t, timestep, i, previous_tube)
                new_position, new_tube = self.move_particle(current_position, current_tube, self.delta_t, timestep, i, previous_tube)
                self.particles_evolution[i, timestep + 2] = new_position
                self.particles_evolution[i, 1] = new_tube
            # else:
            #     self.particles_evolution[i, timestep + 2] = current_position

    def move_particle(self, current_position, current_tube, remaining_time, timestep, particle_index, previous_tube=None):
        if self.rbc_velocity[current_tube] > 0:
            direction = self._calculate_direction_vector(current_tube)
            velocity = self.rbc_velocity[current_tube]
            return self._move_particle_forward(current_position, current_tube, remaining_time, timestep, particle_index, direction, velocity)
        else:
            direction = self._calculate_direction_vector(current_tube)
            velocity = self.rbc_velocity[current_tube]
            return self._move_particle_backward(current_position, current_tube, remaining_time, timestep, particle_index, direction, velocity)

    def _calculate_direction_vector(self, tube_index):
        """Calculate the direction vector for a given tube."""
        y, x = self._calculate_edge_direction(tube_index)
        direction = np.array([x, y, 0]) / np.linalg.norm([x, y, 0])
        return direction

    def _move_particle_forward(self, current_position, current_tube, remaining_time, timestep, particle_index, direction, velocity):
        """Move particle forward within or between tubes."""
        distance_to_travel = velocity * remaining_time
        distance_remaining_in_tube = self._calculate_distance_remaining(current_position, current_tube)

        if distance_to_travel < distance_remaining_in_tube:
            new_position = current_position + direction * distance_to_travel
            return new_position, current_tube
        else:
            return self._handle_tube_transition(current_position, current_tube, remaining_time, timestep, particle_index, velocity, forward=True)

    def _move_particle_backward(self, current_position, current_tube, remaining_time, timestep, particle_index, direction, velocity):
        """Move particle backward within or between tubes."""
        distance_to_travel = velocity * remaining_time
        distance_remaining_in_tube = self._calculate_distance_remaining(current_position, current_tube, forward=False)

        if abs(distance_to_travel) < distance_remaining_in_tube:
            new_position = current_position + direction * distance_to_travel
            return new_position, current_tube
        else:
            return self._handle_tube_transition(current_position, current_tube, remaining_time, timestep, particle_index, velocity, forward=False)

    def _calculate_distance_remaining(self, current_position, current_tube, forward=True):
        """Calculate remaining distance within a tube."""
        if forward:
            return self.length[current_tube] - np.linalg.norm(current_position - self.vs_coords[self.es[current_tube, 0]])
        else:
            return self.length[current_tube] - np.linalg.norm(current_position - self.vs_coords[self.es[current_tube, 1]])

    def _calculate_direction_vector(self, tube_index):
        """Calculate the direction vector for a given tube."""
        y, x = self._calculate_edge_direction(tube_index)
        direction = np.array([x, y, 0]) / np.linalg.norm([x, y, 0])
        return direction

    def _handle_tube_transition(self, current_position, current_tube, remaining_time, timestep, particle_index, velocity, forward=True):
        """Handle the transition of a particle between tubes."""
        if forward:
            new_position = self.vs_coords[self.es[current_tube, 1]]
            remaining_time -= self._calculate_distance_remaining(current_position, current_tube) / velocity
            new_node = self.es[current_tube, 1]
        else:
            new_position = self.vs_coords[self.es[current_tube, 0]]
            remaining_time -= self._calculate_distance_remaining(current_position, current_tube, forward=False) / velocity
            new_node = self.es[current_tube, 0]

        if new_node in self.outflow_vertices:
            self.particles_evolution[particle_index, timestep + 2:] = np.nan
            return new_position, current_tube

        valid_edges = self._find_valid_edges(new_node, current_tube)

        if len(valid_edges) > 0:
            selected_edge = self._select_next_edge(valid_edges)
            return self.move_particle(new_position, selected_edge, remaining_time, timestep, particle_index)
        else:
            return new_position, current_tube
            
    def _select_next_edge(self, valid_edges):
        """Select the next edge for a particle based on flow rates."""
        total_flow_rate = sum(abs(self.flow_rate[e]) for e in valid_edges)
        probabilities = [abs(self.flow_rate[e]) / total_flow_rate for e in valid_edges]
        selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]
        return selected_edge

    def animate_particles(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Variables para límites de los ejes
        x_min = min(self.vs_coords[:, 0].min(), self.initial_particles_coords[:, 0].min())
        x_max = max(self.vs_coords[:, 0].max(), self.initial_particles_coords[:, 0].max())
        y_min = min(self.vs_coords[:, 1].min(), self.initial_particles_coords[:, 1].min())
        y_max = max(self.vs_coords[:, 1].max(), self.initial_particles_coords[:, 1].max())

        # Ajustar los valores de los límites de la figura
        margin = 0.1 * (x_max - x_min)
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin

        # Variable para mantener la animación en memoria
        self.ani = None

        def animate(t):
            ax.clear()  # Limpiar la trama en cada cuadro

            # Redibujar la red
            ig.plot(
                self.graph,
                target=ax,
                layout=self.vs_coords[:, :2],
                vertex_size=20,
                vertex_color="white",
                vertex_label=[f"v{v}" for v in range(len(self.vs_coords))],
                vertex_label_dist=1.5,
                edge_color="red",
                edge_width=4,
                edge_label=[f"e{e}" for e in range(len(self.es))],
            )

            # Dibujar partículas en la posición actual
            for i in range(len(self.particles_evolution)):
                current_position = self.particles_evolution[i, t + 2]
                if isinstance(current_position, np.ndarray) and not np.isnan(current_position).any():
                    ax.scatter(
                        current_position[0],  # x-coord
                        current_position[1],  # y-coord
                        color='black', s=50, zorder=5
                    )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'Timestep: {t}')

        def start_animation():
            self.ani = animation.FuncAnimation(fig, animate, frames=self.N_timesteps + 1, interval=500, repeat=False)
            plt.draw()

        start_animation()

        def restart(event):
            if self.ani is not None:
                self.ani.event_source.stop()  # Detener la animación actual
            start_animation()

        ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
        button = Button(ax_button, 'Restart')
        button.on_clicked(restart)

        plt.show()
