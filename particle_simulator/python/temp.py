import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os
from matplotlib.animation import FuncAnimation

# Parameters for controlling simulation
dt = 0.5  # Time step
dx = 1.0  # Spatial step
T = 100.0  # Total simulation time
D = 0.95  # Diffusion coefficient
k = 1000  # Chemotactic sensitivity
chemo_source = (50, 50)  # Location of chemoattractant source
# Get the absolute path of the directory containing the current script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Define the relative paths from the script directory
output_folder_position = os.path.join(script_directory, 'scatter_plots')
output_folder_concentration = os.path.join(script_directory, 'concentration')
# Parameters for controlling Post processing
PLOT_POSITIONS = False
PLOT_CONCENTRATION = False
MAKE_GIF = True

# Grid size and initialization
grid_size = 100


def make_gif(output_folder):
    print("Generating Animation...")
    # Get the list of image files in the directory
    try:
        images = [img for img in os.listdir(
            output_folder) if img.endswith(".jpg")]
        # Sort the images based on time steps
        images.sort(key=lambda x: float(x.split('_')[1].split('.jpg')[0]))
    except:
        assert "Folder doesnt exist or images are not plotted."

    def update(frame):
        plt.clf()  # Clear the previous plot
        image_path = os.path.join(output_folder, images[frame])
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.title(
            f'Time Step: {float(images[frame].split("_")[1].split(".jpg")[0])}')
        plt.axis('off')

    # Set up the figure
    fig, ax = plt.subplots()

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(
        images), interval=200, blit=False)

    # Specify the name of the output animation file
    animation_file = 'output_animation.gif'

    # Save the animation
    animation.save(animation_file, writer='pillow')


def update_bacteria_position(current_concentration, bacteria_positions):
    # print(current_concentration)
    bacteria_positions_updated = []
    for x, y in bacteria_positions:
        # print(x, y)
        gradient_x = -(current_concentration[(x + 1) %
                                             grid_size, y] - current_concentration[x, y]) / dx
        gradient_y = -(current_concentration[x, (y + 1) %
                                             grid_size] - current_concentration[x, y]) / dx
        move_x = k * gradient_x
        move_y = k * gradient_y
        xold = x
        yold = y
        x += int(move_x * dt)
        y += int(move_y * dt)
        x = x % grid_size
        y = y % grid_size
        if x < 100 and y < 100:
            bacteria_positions.append((x, y))
            bacteria_positions.remove((xold, yold))
    return bacteria_positions


def plot_concentration_field(concentration, output_folder_concentration, t):
    os.makedirs(output_folder_concentration, exist_ok=True)
    # Display the concentration field
    plt.imshow(concentration, cmap='hot', origin='lower',
               extent=[0, grid_size, 0, grid_size])
    plt.colorbar(label='Chemotactic Concentration')
    plt.title('Keller-Segel Model Simulation')
    plt.savefig(f"{output_folder_concentration}/fig_{t}.jpg")
    plt.close()


def plot_positions(bacteria, t):
    assert len(
        bacteria) == 100, "Number of bacteria decreased. Bacteria are dying!"
    os.makedirs(output_folder_position, exist_ok=True)
    # Create a scatter plot
    # Unpack the pairs into separate lists for x and y coordinates
    x1, y1 = zip(*bacteria)
    # Path to the directory to save the scatter plot images
    # Create a 100x100 grid
    grid11_size = 100
    plt.figure(figsize=(6, 6))  # Set the figure size

    # Plot the points on the grid
    # Adjust marker and color as needed
    plt.scatter(x1, y1, marker='o', s=10, c='blue')

    # Set axis limits to match the grid size
    plt.xlim(0, grid11_size)
    plt.ylim(0, grid11_size)

    # Optionally, add labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Scatter Plot on a 100x100 Grid")
    plt.savefig(f"{output_folder_position}/fig_{t}.jpg")
    plt.close()
    # plt.show()


def update_concentration(concentration):
    new_concentration = np.zeros((grid_size, grid_size))
    for x in range(grid_size):
        for y in range(grid_size):
            laplacian = (concentration[(x + 1) % grid_size, y] + concentration[(x - 1) % grid_size, y]
                         + concentration[x, (y + 1) % grid_size] +
                         concentration[x, (y - 1) % grid_size]
                         - 4 * concentration[x, y]) / (dx ** 2)
            new_concentration[x, y] = concentration[x, y] + D * laplacian * dt
    new_concentration[50, 50] = 1
    return new_concentration


def reset_concentration():
    concentration = np.zeros((grid_size, grid_size))
    concentration[50, 50] = 1
    return concentration


def simulate(concentration, bacteria_positions):
    # Time-stepping loop
    print("Starting Simulation:\n")
    for t in np.arange(0, T, dt):
        print("Time step: ", int(t/dt))
        bacteria_positions = update_bacteria_position(
            concentration, bacteria_positions)
        if PLOT_POSITIONS:
            plot_positions(bacteria_positions, t)
        concentration = update_concentration(
            concentration)
        if PLOT_CONCENTRATION:
            plot_concentration_field(
                concentration, output_folder_concentration, t)


if __name__ == '__main__':
    #initialization
    concentration = np.zeros((grid_size, grid_size))
    concentration[50, 50] = 1
    bacteria = [(random.randint(0, grid_size - 1),
                 random.randint(0, grid_size - 1)) for _ in range(100)]
    #main simulation
    simulate(concentration, bacteria)
    #Postprocessing
    if MAKE_GIF is True:
        if PLOT_POSITIONS:
            make_gif(output_folder_position)
        if PLOT_CONCENTRATION:    
            make_gif(output_folder_concentration)
