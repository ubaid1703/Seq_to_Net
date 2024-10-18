import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
import jax
from jax_md import space, minimize, energy
from typing import Dict, Tuple, List, Union
import dataclasses
from enum import Enum


class AminoAcid(Enum):
    A = 0  # Alanine
    B = 1  # Lets say B is for a different amino acid
 

# Update the constants
SIGMA_VALUES: np.ndarray = np.array([
    [0.5, 0.55],
    [0.55, 0.6]   
])

EPSILON_VALUES: np.ndarray = np.array([
    [1.0, 1.0],
    [1.0, 1.0]
])

AMINO_ACID_COLORS: List[str] = ['red', 'blue', 'green', 'purple']

class PlottingMethod(str, Enum):
    SNAKING = 'snaking'
    SPIRAL = 'spiral'
    CONSTANT_SPIRAL = 'constant_spiral'

# Define simulation parameters
@dataclasses.dataclass
class SimulationParams:
    box_size: float = 10.0
    num_steps: int = 3000
    print_interval: int = 1
    animation_interval: int = 3
    initial_scale: float = 0.8
    alpha: int = 2
    bond_stiffness: float = 2.0
    soft_sphere_scale: float = 0.05
    spiral_turn_ratio: float = 2 * np.pi * (2 - np.sqrt(3))
    plotting_method: PlottingMethod = PlottingMethod.CONSTANT_SPIRAL
    snake_line_length: int = 5 
    initial_position_display_time: float = 3.0  # Time in seconds to display initial position


@jax.jit
def custom_spring_bond(r, length, k):
    """Custom spring bond energy function."""
    return 0.5 * k * (r - length)**2

def create_system_potential(sequence, params, displacement_fn, pairwise_displacement_fn):
    """Create a closure for the system potential function."""
    
    # Pre-compute bond lengths
    n = len(sequence)
    bond_lengths = jnp.array([(SIGMA_VALUES[sequence[i]][sequence[i]] + SIGMA_VALUES[sequence[i+1]][sequence[i+1]]) / 2 for i in range(n-1)])
    
    # Convert sequence to a JAX array
    sequence_array = jnp.array(sequence)
    
    @jax.jit
    def system_potential(positions):
        """Calculate the total system potential energy."""
        dR = pairwise_displacement_fn(positions, positions)
        dr = space.distance(dR)
        
        sigma = jnp.take(SIGMA_VALUES, sequence_array, axis=0)
        sigma = jnp.take(sigma, sequence_array, axis=1)
        epsilon = jnp.take(EPSILON_VALUES, sequence_array, axis=0)
        epsilon = jnp.take(epsilon, sequence_array, axis=1)
        
        pairwise_energies = energy.soft_sphere(dr, sigma=sigma, epsilon=epsilon * params.soft_sphere_scale, alpha=params.alpha)
        soft_sphere_energy = (jnp.sum(jnp.triu(pairwise_energies, k=1)) * 2 + 
                             jnp.sum(jnp.diag(pairwise_energies)))
        
        # Spring bond energies with pre-computed bond lengths
        bonds = jnp.array([(i, i+1) for i in range(n-1)])
        
        vectorized_displacement = jax.vmap(displacement_fn)
        bond_vectors = vectorized_displacement(positions[bonds[:, 0]], positions[bonds[:, 1]])
        
        actual_bond_lengths = space.distance(bond_vectors)
        bond_energy = jnp.sum(custom_spring_bond(actual_bond_lengths, bond_lengths, params.bond_stiffness))
        
        total_energy = soft_sphere_energy + bond_energy
        
        return total_energy
    
    return system_potential

def generate_initial_positions(n: int, params: SimulationParams, sequence: List[int]) -> jnp.ndarray:
    if params.plotting_method == PlottingMethod.SNAKING:
        nodes = snaking_curve(n, params.snake_line_length)
    elif params.plotting_method == PlottingMethod.SPIRAL:
        nodes = spiral_curve(n, params.spiral_turn_ratio)
    elif params.plotting_method == PlottingMethod.CONSTANT_SPIRAL:
        nodes = constant_spiral_curve(n, sequence)
    else:
        raise ValueError(f"Invalid plotting method: {params.plotting_method}")
    
    return jnp.array(nodes, dtype=jnp.float32) * params.initial_scale

def snaking_curve(n: int, line_length: int) -> np.ndarray:
    points = []
    for i in range(n):
        row = i // line_length
        col = i % line_length if row % 2 == 0 else line_length - (i % line_length) - 1
        points.append([col, row])
    return np.array(points)

def spiral_curve(n: int, turn_ratio: float, center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    points = []
    for i in range(n):
        angle = i * turn_ratio
        radius = np.sqrt(i + 1) * 0.5
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def constant_spiral_curve(n: int, sequence: List[int], center: Tuple[float, float] = (0, 0)) -> np.ndarray:
    points = []
    angle = 0
    radius = 0.5  # Start with a small radius
    
    for i in range(n):
        if i > 0:
            # Calculate the equilibrium bond distance
            bond_length = (SIGMA_VALUES[sequence[i-1]][sequence[i-1]] + SIGMA_VALUES[sequence[i]][sequence[i]]) / 2
            
            # Calculate the turn ratio based on the current radius
            turn_ratio = bond_length / radius
            
            # Update the angle
            angle += turn_ratio
        
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append([x, y])
        
        # Increase the radius slightly for the next point
        radius += 0.01  # may need to change this value for optimal results
    
    return np.array(points)

def plot_positions(positions: Union[np.ndarray, jnp.ndarray], sequence: List[int], filename: str, energy: float, draw_bonds: bool = True):
    plt.figure(figsize=(10, 10))
    positions_np = np.array(positions)
    
    if draw_bonds:
        for i in range(len(sequence) - 1):
            plt.plot([positions_np[i, 0], positions_np[i+1, 0]], 
                     [positions_np[i, 1], positions_np[i+1, 1]], 
                     'k-', alpha=0.3)
    
    for i, (x, y) in enumerate(positions_np):
        color = AMINO_ACID_COLORS[sequence[i]]
        sigma = SIGMA_VALUES[sequence[i]][sequence[i]]
        circle = plt.Circle((x, y), sigma/2, facecolor=color, edgecolor='black', alpha=0.7)
        plt.gca().add_artist(circle)
    
    x_min, x_max = np.min(positions_np[:, 0]), np.max(positions_np[:, 0])
    y_min, y_max = np.min(positions_np[:, 1]), np.max(positions_np[:, 1])
    
    if np.isnan(x_min) or np.isnan(x_max) or np.isnan(y_min) or np.isnan(y_max) or \
       np.isinf(x_min) or np.isinf(x_max) or np.isinf(y_min) or np.isinf(y_max):
        print("Warning: NaN or Inf detected in position data. Using default plot limits.")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
    else:
        max_sigma = np.max(SIGMA_VALUES)
        padding = max_sigma + 0.2
        plt.xlim(x_min - padding, x_max + padding)
        plt.ylim(y_min - padding, y_max + padding)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Energy: {energy:.2f}')
    plt.savefig(filename)
    plt.close()

def plot_energy_trajectory(energy_trajectory: List[float], params: SimulationParams):
    plt.figure(figsize=(10, 6))
    steps = np.arange(0, len(energy_trajectory) * params.print_interval, params.print_interval)
    plt.plot(steps, energy_trajectory)
    plt.xlabel('Optimization Step')
    plt.ylabel('Energy')
    plt.title('Energy vs. Optimization Step')
    plt.yscale('log')  # Use log scale for y-axis as energy values can span multiple orders of magnitude
    plt.grid(True)
    plt.savefig("energy_trajectory.png")
    plt.close()

def run_simulation(sequence: List[int], params: SimulationParams) -> Tuple[jnp.ndarray, List[float], List[jnp.ndarray]]:
    displacement_fn, shift_fn = space.free()
    pairwise_displacement_fn = space.map_product(displacement_fn)

    # Fix: Pass the sequence argument to generate_initial_positions
    initial_positions = generate_initial_positions(len(sequence), params, sequence)
    
    system_potential = create_system_potential(
        jnp.array(sequence),
        params,
        displacement_fn,
        pairwise_displacement_fn
    )
    
    # Calculate initial energy
    initial_energy = float(system_potential(initial_positions))
    plot_positions(initial_positions, sequence, "initial_positions.png", initial_energy)
    
    init_fn, apply_fn = minimize.fire_descent(system_potential, shift_fn)
    state = init_fn(initial_positions)
    
    energy_trajectory = [initial_energy]
    position_trajectory = [np.array(initial_positions)]
    
    for i in range(params.num_steps):
        try:
            state = apply_fn(state)
            if i % params.animation_interval == 0:
                position_trajectory.append(np.array(state.position))
            if i % params.print_interval == 0:
                energy_val = system_potential(state.position)
                energy_trajectory.append(float(energy_val))
                print(f"Step {i}, Energy: {energy_val}")
                
                # Debug output
                energy_val_np = np.array(energy_val)
                if np.isnan(energy_val_np) or np.isinf(energy_val_np):
                    print(f"NaN or Inf detected at step {i}. Stopping simulation.")
                    print(f"Positions: {state.position}")
                    break
        except Exception as e:
            print(f"Error at step {i}: {str(e)}")
            break
    
    final_positions = state.position
    final_energy = float(system_potential(final_positions))
    plot_positions(final_positions, sequence, "final_minimized_positions.png", final_energy, draw_bonds=True)
    return final_positions, energy_trajectory, position_trajectory

def create_animation(position_trajectory: List[np.ndarray], sequence: List[int], params: SimulationParams):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    all_positions = np.concatenate(position_trajectory)
    x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
    y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
    max_sigma = np.max(SIGMA_VALUES)
    padding = max_sigma + 0.2
    
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal', adjustable='box')
    
    circles = []
    for i in range(len(sequence)):
        circle = plt.Circle((0, 0), SIGMA_VALUES[sequence[i]][sequence[i]]/2, facecolor=AMINO_ACID_COLORS[sequence[i]], edgecolor='black', alpha=0.7)
        circles.append(circle)
        ax.add_artist(circle)
    
    line, = ax.plot([], [], 'k-', alpha=0.3, linewidth=1)
    title = ax.set_title("")
    
    fps = 15
    initial_frames = int(params.initial_position_display_time * fps)
    
    def animate(frame):
        if frame < initial_frames:
            positions = position_trajectory[0]
            step = 0
        else:
            positions = position_trajectory[min(frame - initial_frames, len(position_trajectory) - 1)]
            step = (frame - initial_frames) * params.animation_interval
        
        for i, (x, y) in enumerate(positions):
            circles[i].center = (x, y)
        
        bonds_x = np.array([[positions[i, 0], positions[i+1, 0]] for i in range(len(positions)-1)]).flatten()
        bonds_y = np.array([[positions[i, 1], positions[i+1, 1]] for i in range(len(positions)-1)]).flatten()
        line.set_data(bonds_x, bonds_y)
        
        if frame < initial_frames:
            title.set_text(f'Initial Position')
        else:
            title.set_text(f'Optimization Step: {step}')
        
        return circles + [line, title]
    
    num_frames = len(position_trajectory) + initial_frames
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps, blit=True)
    anim.save('optimization_animation.gif', writer='pillow', fps=fps, dpi=100)
    plt.close(fig)

def main():
    # Define the sequence of amino acids
    #sequence_string = "ABBABA"  # ordered
    #sequence = [AminoAcid[aa].value for aa in sequence_string]
    sequence = [AminoAcid.A.value] * 500  #a bunch of one type 
    params = SimulationParams()  
    final_positions, energy_trajectory, position_trajectory = run_simulation(sequence, params)
    print(f"Initial Energy: {energy_trajectory[0]:.2f}")
    print(f"Final Energy: {energy_trajectory[-1]:.2f}")
    plot_energy_trajectory(energy_trajectory, params)
    create_animation(position_trajectory, sequence, params)

if __name__ == "__main__":
    main()
