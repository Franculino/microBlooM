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

        
        self.graph = ig.Graph()  # Inicializar el grafo
        self.graph.add_vertices(self.vs_coords.shape[0])  # Agregar vértices
        self.graph.add_edges(self.es)  # Agregar aristas desde edge_list

        # Asignar atributos a los vértices y aristas
        self.graph.vs['xyz'] = self.vs_coords.tolist()  # Coordenadas de los vértices
        self.graph.es['length'] = self.length  # Longitud de las aristas
        self.graph.es['diameter'] = self.diameter  # Diámetro de las aristas
        self.graph.es['flow_rate'] = self.flow_rate  # Flujo en las aristas
        self.graph.es['rbc_velocity'] = self.rbc_velocity

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

        for i in range(self.N_particles):
            y = self.vs_coords[self.es[self.initial_particle_tube[i], 1], 1] - self.vs_coords[self.es[self.initial_particle_tube[i], 0], 1]
            x = self.vs_coords[self.es[self.initial_particle_tube[i], 1], 0] - self.vs_coords[self.es[self.initial_particle_tube[i], 0], 0]
            delta_x = self.initial_local_coord[i] * self.length[self.initial_particle_tube[i]] * np.cos(np.arctan2(y, x))
            delta_y = self.initial_local_coord[i] * self.length[self.initial_particle_tube[i]] * np.sin(np.arctan2(y, x))
            self.initial_particles_coords[i] = self.vs_coords[self.es[self.initial_particle_tube[i], 0]] + [delta_x, delta_y, 0]
            self.particles_evolution[i, 2] = self.initial_particles_coords[i]
    
    def move_particle(self, current_position, current_tube, remaining_time, timestep, particle_index, previous_tube=None):
        if self.rbc_velocity[current_tube] > 0:
            # Calculate the direction of movement within the vessel
            y = self.vs_coords[self.es[current_tube, 1], 1] - self.vs_coords[self.es[current_tube, 0], 1]
            x = self.vs_coords[self.es[current_tube, 1], 0] - self.vs_coords[self.es[current_tube, 0], 0]
            direction = np.array([x, y, 0]) / np.linalg.norm([x, y, 0])
            
            # Calculate the distance the particle will travel during the remaining time
            velocity = self.rbc_velocity[current_tube]
            distance_to_travel = velocity * remaining_time
            distance_remaining_in_tube = self.length[current_tube] - np.linalg.norm(current_position - self.vs_coords[self.es[current_tube, 0]])
            
            if distance_to_travel < distance_remaining_in_tube:
                # If the particle stays in the same vessel
                new_position = current_position + direction * distance_to_travel
                return new_position, current_tube
            else:
                # If the particle reaches the end of the vessel
                remaining_time -= distance_remaining_in_tube / velocity
                new_position = self.vs_coords[self.es[current_tube, 1]]

                if self.es[current_tube, 1] in self.outflow_vertices:
                    # Rellenar con np.nan y detener la actualización
                    self.particles_evolution[particle_index, timestep + 2:] = np.nan
                    return new_position, current_tube
                
                # Find the edges connected to the new node
                connected_edges = self.graph.incident(self.es[current_tube, 1], mode="OUT")
                connected_edges = [e for e in connected_edges if e != current_tube]

                valid_edges = []
            
                for e in connected_edges:
                    node1, node2 = self.es[e]
                
                    # Determine direction of the flow
                    if self.es[current_tube, 1] == node1 and self.rbc_velocity[e] > 0:
                        valid_edges.append(e)
                    elif self.es[current_tube, 1] == node2 and self.rbc_velocity[e] < 0:
                        valid_edges.append(e)
                valid_edges = np.array(valid_edges)

                if len(valid_edges) > 0:
                    total_flow_rate = sum(abs(self.flow_rate[e]) for e in valid_edges)
                    probabilities = [abs(self.flow_rate[e]) / total_flow_rate for e in valid_edges]
                    selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]
                    return self.move_particle(new_position, selected_edge, remaining_time, timestep, particle_index)
                else:
                    # If no more connected vessels, the particle stops here
                    return new_position, current_tube
        else:
            # Código para la dirección opuesta (flujo inverso)
            y = self.vs_coords[self.es[current_tube, 1], 1] - self.vs_coords[self.es[current_tube, 0], 1]
            x = self.vs_coords[self.es[current_tube, 1], 0] - self.vs_coords[self.es[current_tube, 0], 0]
            direction = np.array([x, y, 0]) / np.linalg.norm([x, y, 0])
            
            velocity = self.rbc_velocity[current_tube]
            distance_to_travel = velocity * remaining_time
            distance_remaining_in_tube = self.length[current_tube] - np.linalg.norm(current_position - self.vs_coords[self.es[current_tube, 1]])
            
            if abs(distance_to_travel) < distance_remaining_in_tube:
                new_position = current_position + direction * distance_to_travel
                return new_position, current_tube
            else:
                remaining_time -= distance_remaining_in_tube / velocity
                new_position = self.vs_coords[self.es[current_tube, 0]]

                if self.es[current_tube, 1] in self.outflow_vertices:
                    self.particles_evolution[particle_index, timestep + 2:] = np.nan
                    return new_position, current_tube
                
                connected_edges = self.graph.incident(self.es[current_tube, 0], mode="OUT")
                connected_edges = [e for e in connected_edges if e != current_tube]

                valid_edges = []
            
                for e in connected_edges:
                    node1, node2 = self.es[e]
                
                    if self.es[current_tube, 0] == node1 and self.rbc_velocity[e] > 0:
                        valid_edges.append(e)
                    elif self.es[current_tube, 0] == node2 and self.rbc_velocity[e] < 0:
                        valid_edges.append(e)
                valid_edges = np.array(valid_edges)

                if len(valid_edges) > 0:
                    total_flow_rate = sum(abs(self.flow_rate[e]) for e in valid_edges)
                    probabilities = [abs(self.flow_rate[e]) / total_flow_rate for e in valid_edges]
                    selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]
                    return self.move_particle(new_position, selected_edge, remaining_time, timestep, particle_index)
                else:
                    return new_position, current_tube

    def simulate(self):
        for t in range(1, self.N_timesteps + 1):
            for inflow_vertex, interval in self.inflow_vertices.items():
                if t % interval == 0:
                    connected_edges = self.graph.incident(inflow_vertex, mode="OUT")

                    valid_edges = []
                
                    for e in connected_edges:
                        node1, node2 = self.es[e]
                        if inflow_vertex == node1 and self.rbc_velocity[e] > 0:
                            valid_edges.append(e)
                        elif inflow_vertex == node2 and self.rbc_velocity[e] < 0:
                            valid_edges.append(e)
                    valid_edges = np.array(valid_edges)

                    if len(valid_edges) > 0:
                        total_flow_rate = sum(abs(self.flow_rate[e]) for e in valid_edges)
                        probabilities = [abs(self.flow_rate[e]) / total_flow_rate for e in valid_edges]
                        selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]

                        y = self.vs_coords[self.es[selected_edge, 1], 1] - self.vs_coords[self.es[selected_edge, 0], 1]
                        x = self.vs_coords[self.es[selected_edge, 1], 0] - self.vs_coords[self.es[selected_edge, 0], 0]
                        new_position = self.vs_coords[self.es[selected_edge, 0]] + np.array([x, y, 0]) * 1e-09

                        new_particle_index = len(self.particles_evolution)
                        new_row = np.full((1, self.N_timesteps + 3), np.nan, dtype=object)
                        new_row[0, 0] = new_particle_index
                        new_row[0, 1] = selected_edge
                        new_row[0, 2 + t-1] = new_position
                        new_row[0, 2 + t:] = 0
                        self.particles_evolution = np.vstack([self.particles_evolution, new_row])

            for i in range(len(self.particles_evolution)):
                current_position = self.particles_evolution[i, t + 2 - 1]
                current_tube = self.particles_evolution[i, 1]
                previous_tube = current_tube

                if isinstance(current_position, np.ndarray):
                    new_position, new_tube = self.move_particle(current_position, current_tube, self.delta_t, t, i, previous_tube)
                    self.particles_evolution[i, t + 2] = new_position
                    self.particles_evolution[i, 1] = new_tube
                else:
                    self.particles_evolution[i, t + 2] = current_position

        return self.particles_evolution
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
