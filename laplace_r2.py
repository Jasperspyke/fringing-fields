# prompt: plot ax2 upside down
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.animation import FuncAnimation, PillowWriter


class Particle:
    def __init__(self, location, Ex, Ey):
        self.location = location  # (y, x) tuple
        self.Ex = Ex
        self.Ey = Ey

    def field(self):
        y, x = self.location
        return self.Ex[y, x], self.Ey[y, x]

def initialize_grid(size=150):
    """Initialize the grid with boundary conditions."""
    grid = np.zeros((size, size))
    grid[size//3, size//4:3*size//4] = 1
    grid[2*size//3, size//4:3*size//4] = -1
    return grid

def solve_laplace(grid, iterations=1000, tolerance=1e-4):
    """Solve Laplace equation using five-point stencil method."""
    size = len(grid)
    for _ in range(iterations):
        old_grid = grid.copy()
        for i in range(1, size-1):
            for j in range(1, size-1):
                if old_grid[i, j] in [1, -1]:
                    continue
                grid[i, j] = 0.25 * (
                    old_grid[i+1, j] + old_grid[i-1, j] +
                    old_grid[i, j+1] + old_grid[i, j-1]
                )
        if np.max(np.abs(grid - old_grid)) < tolerance:
            break
    return grid

def calculate_field(potential):
    """Calculate electric field components from potential."""
    Ey, Ex = np.gradient(-potential)
    return Ex, Ey


def plot_results(frame, potential, Ex, Ey, particle_list, fig, ax1, ax2):
    """Update plot for animation."""
    ax1.clear()
    ax2.clear()

    # Plot potential
    im1 = ax1.imshow(potential, cmap='RdBu')
    ax1.set_title('Electric Potential')

    # Plot field vectors
    skip = 5
    y, x = np.mgrid[0:potential.shape[0]:skip, 0:potential.shape[1]:skip]
    E_magnitude = np.sqrt(Ex[::skip, ::skip] ** 2 + Ey[::skip, ::skip] ** 2)
    Ex_norm = Ex[::skip, ::skip] / (E_magnitude + 1e-10)
    Ey_norm = Ey[::skip, ::skip] / (E_magnitude + 1e-10)
    ax2.quiver(x, y, Ex_norm, Ey_norm)
    ax2.set_title('Electric Field Lines')
    ax2.invert_yaxis()  # Invert y-axis for ax2

    # Plot particle on both axes
    for particle in particle_list:
        px, py = particle.location[1], particle.location[0]
        ax1.plot(px, py, 'ro', markersize=6)
        ax2.plot(px, py, 'ro', markersize=6)
        # Overlay field vector at particle location
        fx, fy = particle.field()
        fx *= 2
        fy *= 2
        ax1.quiver(px, py, fx, fy, color='yellow', scale=5)
        ax2.quiver(px, py, fx, fy, color='yellow', scale=5)

    # Update particles
    for i in range(len(particle_list)):
        particle_list[i] = particle_next(particle_list[i], Ex, Ey)

    plt.tight_layout()


def update_particles(particle_list, Ex, Ey):
    """Update particle locations and get new field vectors."""
    new_particles = []
    for particle in particle_list:
        new_particle = particle_next(particle, Ex, Ey)
        new_particles.append(new_particle)
    return new_particles

def particle_next(particle, Ex, Ey):
    """Update particle location and get the new field vector."""
    y = int(particle.location[0])  # Ensure indices are integers
    x = int(particle.location[1])  # Ensure indices are integers
    uy, ux = particle.field()  # Get the field at the current location
    new_y = int(y + ux * 100)  # Update y by field component
    new_x = int(x + uy * 100)  # Update x by field component

    # Ensure the particle location is within the grid bounds
    new_y = np.clip(new_y, 0, len(Ex) - 1)
    new_x = np.clip(new_x, 0, len(Ex[0]) - 1)

    particle.location = (new_y, new_x)

    return particle


def main():
    grid = initialize_grid()
    potential = solve_laplace(grid)
    Ex, Ey = calculate_field(potential)

    # Initialize 20 particles at random locations
    particle_list = []
    grid_size = len(grid)
    for _ in range(20):
        y = np.random.randint(0, grid_size)
        x = np.random.randint(0, grid_size)
        particle = Particle((y, x), Ex, Ey)
        particle_list.append(particle)

    # Prepare for animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    def animation_step(frame):
        plot_results(frame, potential, Ex, Ey, particle_list, fig, ax1, ax2)

    ani = FuncAnimation(fig, animation_step, frames=100, interval=200)
    writer = PillowWriter(fps=5)  # Set frames per second
    ani.save("animation999.gif", writer=writer)  # Save animation as gif
    plt.show()


if __name__ == "__main__":
    main()

