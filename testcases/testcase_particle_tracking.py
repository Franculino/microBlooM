import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.animation as animation
from matplotlib.widgets import Button
import random

def move_particle(vs_coords, es, current_position, current_tube, remaining_time, outflow_vertices, timestep, particles_evolution, particle_index,  previous_tube=None):
    if rbc_velocity[current_tube] > 0:
        # Calculate the direction of movement within the vessel
        y = vs_coords[es[current_tube, 1], 1] - vs_coords[es[current_tube, 0], 1]
        x = vs_coords[es[current_tube, 1], 0] - vs_coords[es[current_tube, 0], 0]
        direction = np.array([x, y, 0]) / np.linalg.norm([x, y, 0])
        
        # Calculate the distance the particle will travel during the remaining time
        velocity = rbc_velocity[current_tube]
        distance_to_travel = velocity * remaining_time
        distance_remaining_in_tube = length[current_tube] - np.linalg.norm(current_position - vs_coords[es[current_tube, 0]])
        
        if distance_to_travel < distance_remaining_in_tube:
            # If the particle stays in the same vessel
            new_position = current_position + direction * distance_to_travel
            return new_position, current_tube
        else:
            # If the particle reaches the end of the vessel
            remaining_time -= distance_remaining_in_tube / velocity
            new_position = vs_coords[es[current_tube, 1]]

            if es[current_tube, 1] in outflow_vertices:
                # Rellenar con np.nan y detener la actualización
                particles_evolution[particle_index, timestep + 2:] = np.nan
                return new_position, current_tube
            
            # Find the edges connected to the new node
            connected_edges = graph.incident(es[current_tube, 1], mode="OUT")
            connected_edges = [e for e in connected_edges if e != current_tube]

            valid_edges = []
        
            for e in connected_edges:
                node1, node2 = es[e]
            
                # Determine direction of the flow
                if es[current_tube, 1] == node1 and rbc_velocity[e] > 0:
                    valid_edges.append(e)
                elif es[current_tube, 1] == node2 and rbc_velocity[e] < 0:
                    valid_edges.append(e)
            valid_edges = np.array(valid_edges)

            if len(valid_edges) > 0:
                # If there is a bifurcation, select the edge with the highest flow rate
                # max_flow_rate_edge = max(connected_edges, key=lambda e: abs(flow_rate[e]))
                # return move_particle(vs_coords, es, new_position, max_flow_rate_edge, remaining_time, outflow_vertices, timestep, particles_evolution, particle_index)
                
                # Decision made in terms of the probability associated with the flow rate in each vessel
                total_flow_rate = sum(abs(flow_rate[e]) for e in valid_edges)
                probabilities = [abs(flow_rate[e]) / total_flow_rate for e in valid_edges]
                selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]
                return move_particle(vs_coords, es, new_position, selected_edge, remaining_time, outflow_vertices, timestep, particles_evolution, particle_index)
            else:
                # If no more connected vessels, the particle stops here
                return new_position, current_tube
    else:
        # Calculate the direction of movement within the vessel
        y = vs_coords[es[current_tube, 1], 1] - vs_coords[es[current_tube, 0], 1]
        x = vs_coords[es[current_tube, 1], 0] - vs_coords[es[current_tube, 0], 0]
        direction = np.array([x, y, 0]) / np.linalg.norm([x, y, 0])
        
        # Calculate the distance the particle will travel during the remaining time
        velocity = rbc_velocity[current_tube]
        distance_to_travel = velocity * remaining_time
        distance_remaining_in_tube = length[current_tube] - np.linalg.norm(current_position - vs_coords[es[current_tube, 1]])
        
        if abs(distance_to_travel) < distance_remaining_in_tube:
            # If the particle stays in the same vessel
            new_position = current_position + direction * distance_to_travel
            return new_position, current_tube
        else:
            # If the particle reaches the end of the vessel
            remaining_time -= distance_remaining_in_tube / velocity
            new_position = vs_coords[es[current_tube, 0]]

            if es[current_tube, 1] in outflow_vertices:
                # Rellenar con np.nan y detener la actualización
                particles_evolution[particle_index, timestep + 2:] = np.nan
                return new_position, current_tube
            
            # Find the edges connected to the new node
            connected_edges = graph.incident(es[current_tube, 0], mode="OUT")
            connected_edges = [e for e in connected_edges if e != current_tube]

            valid_edges = []
        
            for e in connected_edges:
                node1, node2 = es[e]
            
                # Determine direction of the flow
                if es[current_tube, 0] == node1 and rbc_velocity[e] > 0:
                    valid_edges.append(e)
                elif es[current_tube, 0] == node2 and rbc_velocity[e] < 0:
                    valid_edges.append(e)
            valid_edges = np.array(valid_edges)

            if len(valid_edges) > 0:
                # If there is a bifurcation, select the edge with the highest flow rate
                # max_flow_rate_edge = max(connected_edges, key=lambda e: abs(flow_rate[e]))
                # return move_particle(vs_coords, es, new_position, max_flow_rate_edge, remaining_time, outflow_vertices, timestep, particles_evolution, particle_index)
                
                # Decision made in terms of the probability associated with the flow rate in each vessel
                total_flow_rate = sum(abs(flow_rate[e]) for e in valid_edges)
                probabilities = [abs(flow_rate[e]) / total_flow_rate for e in valid_edges]
                selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]
                return move_particle(vs_coords, es, new_position, selected_edge, remaining_time, outflow_vertices, timestep, particles_evolution, particle_index)
            else:
                # If no more connected vessels, the particle stops here
                return new_position, current_tube


graph = ig.Graph.Read_Pickle('testcases\network_simulated3.pkl')
es = np.array(graph.get_edgelist())
vs_coords = np.array(graph.vs['xyz'])
length = np.array(graph.es['length'])
diameter = np.array(graph.es['diameter'])
flow_rate = np.array(graph.es['flow_rate'])
rbc_velocity = np.array(graph.es['rbc_velocity'])

N_particles = 8
initial_particles_coords = np.zeros((N_particles,3))
initial_particle_tube = np.zeros(N_particles)
initial_local_coord = np.full(N_particles, 0.5) # tube reference frame, adimensionalized with length of each tube 
initial_particle_tube = [0,1,9,85,38,42, 70, 32] # np.full(N_particles, 0) #[0,1,12,15]  


for i in np.arange(N_particles):
    y = vs_coords[es[initial_particle_tube[i],1], 1] - vs_coords[es[initial_particle_tube[i],0], 1] 
    x = vs_coords[es[initial_particle_tube[i],1], 0] - vs_coords[es[initial_particle_tube[i],0], 0] 
    delta_x = initial_local_coord[i] * length[initial_particle_tube[i]] * np.cos(np.arctan2(y,x)) 
    delta_y = initial_local_coord[i] * length[initial_particle_tube[i]] * np.sin(np.arctan2(y,x)) 
    initial_particles_coords[i] = vs_coords[es[initial_particle_tube[i],0]] + [delta_x,delta_y,0]

# Define tehe values for adjusting the figure 
x_min = min(vs_coords[:, 0].min(), initial_particles_coords[:, 0].min())
x_max = max(vs_coords[:, 0].max(), initial_particles_coords[:, 0].max())
y_min = min(vs_coords[:, 1].min(), initial_particles_coords[:, 1].min())
y_max = max(vs_coords[:, 1].max(), initial_particles_coords[:, 1].max())

# Adjuste the limit values of the figure
margin = 0.1 * (x_max - x_min)
x_min -= margin
x_max += margin
y_min -= margin
y_max += margin
 
delta_t = length.min()/(rbc_velocity.max())
N_timesteps = 50

edge_colors = ['red' if v > 0 else 'blue' for v in rbc_velocity]
fig, ax = plt.subplots(figsize=(8,8))
ig.plot(
    graph, 
    target=ax,
    layout=vs_coords[:, :2],
    vertex_size=20,
    vertex_color = "white",
    vertex_label=[f"v{v}" for v in range(len(vs_coords))], 
    vertex_label_dist=1.5,  
    edge_color=edge_colors,
    edge_width = 4,
    edge_label=np.array(graph.get_eids(es)),
    # edge_label_dist=10
)

ax.scatter(initial_particles_coords[:, 0], initial_particles_coords[:, 1], color='black', s=50, zorder=5, label='Particles')
ax.legend()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.show()

particles_evolution = np.zeros((N_particles, N_timesteps + 3), dtype=object) # position in each time step, current vessel, initial posion
particles_evolution[:, 0] = np.arange(N_particles)
particles_evolution[:, 1] = initial_particle_tube[:]
for i in range(N_particles):
    particles_evolution[i, 2] = initial_particles_coords[i]

inflow_vertices = {18:3, 54:4, 23:5} # inflow node:number of timesteps between particles
outflow_vertices = [65] # [27]

for t in range(1, N_timesteps + 1):
    for inflow_vertex, interval in inflow_vertices.items():
        if t % interval == 0:
            # Determinar los edges conectados al inflow vertex
            connected_edges = graph.incident(inflow_vertex, mode="OUT")

            valid_edges = []
        
            for e in connected_edges:
                node1, node2 = es[e]
            
                # Determine direction of the flow
                if inflow_vertex == node1 and rbc_velocity[e] > 0:
                    valid_edges.append(e)
                elif inflow_vertex == node2 and rbc_velocity[e] < 0:
                    valid_edges.append(e)
            valid_edges = np.array(valid_edges)

            
            if len(valid_edges) > 0:
                # Calcular la probabilidad basada en el flow rate para cada edge conectado
                total_flow_rate = sum(abs(flow_rate[e]) for e in valid_edges)
                probabilities = [abs(flow_rate[e]) / total_flow_rate for e in valid_edges]

                # Seleccionar un edge basado en estas probabilidades
                selected_edge = random.choices(valid_edges, weights=probabilities, k=1)[0]

                # Configurar la posición inicial de la nueva partícula
                y = vs_coords[es[selected_edge, 1], 1] - vs_coords[es[selected_edge, 0], 1]
                x = vs_coords[es[selected_edge, 1], 0] - vs_coords[es[selected_edge, 0], 0]
                new_position = vs_coords[es[selected_edge, 0]] + np.array([x, y, 0]) *  1e-09

                new_particle_index = len(particles_evolution)
                new_row = np.full((1, N_timesteps + 3), np.nan, dtype=object)
                new_row[0, 0] = new_particle_index  # Asignar el índice de la nueva partícula
                new_row[0, 1] = selected_edge       # Asignar el tubo en el que está
                new_row[0, 2 + t-1] = new_position    # Asignar la posición inicial de la partícula en el timestep actual
                new_row[0, 2 + t:] = 0
                # Añadir la nueva fila a particles_evolution
                particles_evolution = np.vstack([particles_evolution, new_row])

    for i in range(len(particles_evolution)):
        # Get current position and current vessel of the particle
        current_position = particles_evolution[i, t + 2 - 1]  # Access the last recorded position
        current_tube = int(particles_evolution[i, 1])
        
        # Calculate the new position and the new vessel
        new_position, new_tube = move_particle(vs_coords, es, current_position, current_tube, delta_t, outflow_vertices, t, particles_evolution, i)
        
        if not np.isnan(new_position).all():
            particles_evolution[i, t + 2] = new_position

        # # Store the new position in the evolution array
        # particles_evolution[i, t + 2] = new_position
        
        # Update the current vessel ID for the next timestep
        particles_evolution[i, 1] = new_tube

# print('hola')

def animate(t):
    ax.clear()  # Clean the plot in eahc frame
    
    # Redraw the network
    ig.plot(
    graph, 
    target=ax,
    layout=vs_coords[:, :2],
    vertex_size=20,
    vertex_color = "white",
    vertex_label=[f"v{v}" for v in range(len(vs_coords))], 
    vertex_label_dist=1.5,  
    edge_color=edge_colors,
    edge_width = 4,
    edge_label=np.array(graph.get_eids(es)),
    # edge_label_dist=10
    )
    
    # Draw partciles in current position
    for i in range(len(particles_evolution)):
        current_position = particles_evolution[i, t + 2]

        # Verificar si la posición actual es NaN
        if not np.isnan(current_position).any():
            ax.scatter(
                current_position[0],  # x-coord
                current_position[1],  # y-coord
                color='black', s=50, zorder=5
            )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'Timestep: {t}')

def start_animation():
    global ani
    ani = animation.FuncAnimation(fig, animate, frames=N_timesteps+1, interval=500, repeat=False)
    plt.draw()

fig, ax = plt.subplots(figsize=(8, 8))
start_animation()

# Define the function to restart the animation
def restart(event):
    start_animation()

# Crear un botón para reiniciar la animación
ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])  # Definir la posición del botón
button = Button(ax_button, 'Restart')
button.on_clicked(restart)  # Vincular el botón a la función de reinicio
# Mostrar la animación

# ani.save('mi_animacion.gif', writer='pillow', fps=2)
plt.show()