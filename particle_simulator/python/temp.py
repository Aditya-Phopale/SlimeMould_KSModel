import numpy as np
import matplotlib.pyplot as plt
import random
import copy

# Parameters
dt = 0.1  # Time step
dx = 1.0  # Spatial step
T = 1000.0  # Total simulation time
D = 0.5  # Diffusion coefficient
k = 10  # Chemotactic sensitivity
chemo_source = (50, 50)  # Location of chemoattractant source

# Grid size and initialization
grid_size = 100
concentration = np.zeros((grid_size, grid_size))
# concentration[50,50] = 10
bacteria = [(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)) for _ in range(100)]
bacteria_float = copy.deepcopy(bacteria)

# Time-stepping loop
for t in np.arange(0, T, dt):
    new_concentration = np.zeros((grid_size, grid_size))
    for x, y in bacteria:
        x_float = x
        y_float = y

        gradient_x = (concentration[(x + 1) % grid_size, y] - concentration[x, y]) / dx
        gradient_y = (concentration[x, (y + 1) % grid_size] - concentration[x, y]) / dx
        if gradient_y > 0 or gradient_x > 0:
            print(x, " ", y, " ", t)
        # print(gradient_x, " ", gradient_y)
        move_x = k * gradient_x
        move_y = k * gradient_y
        
        x += int(move_x * dt)
        y += int(move_y * dt)
        x_float += move_x * dt
        y_float += move_y * dt
        # print(bacteria)
        # print(bacteria_float)
        bacteria.remove((x, y))
        bacteria_float.remove((x_float, y_float))
        x = x % grid_size
        y = y % grid_size
        x_float = x_float % grid_size
        y_float = y_float % grid_size

        bacteria.append((x, y))
        bacteria_float.append((x_float, y_float))
    for x_float, y_float in bacteria_float:
        
        gradient_x = (concentration[(x + 1) % grid_size, y] - concentration[x, y]) / dx
        gradient_y = (concentration[x, (y + 1) % grid_size] - concentration[x, y]) / dx
        if gradient_y > 0 or gradient_x > 0:
            print(x, " ", y, " ", t)
        # print(gradient_x, " ", gradient_y)
        move_x = k * gradient_x
        move_y = k * gradient_y
        
        x += int(move_x * dt)
        y += int(move_y * dt)
        x_float += move_x * dt
        y_float += move_y * dt
        # print(bacteria)
        # print(bacteria_float)
        bacteria.remove((x, y))
        bacteria_float.remove((x_float, y_float))
        x = x % grid_size
        y = y % grid_size
        x_float = x_float % grid_size
        y_float = y_float % grid_size

        bacteria.append((x, y))
        bacteria_float.append((x_float, y_float))
    
    # Create a scatter plot
    x1, y1 = zip(*bacteria_float)  # Unpack the pairs into separate lists for x and y coordinates

    # Create a 100x100 grid
    grid11_size = 100
    plt.figure(figsize=(6, 6))  # Set the figure size

    # Plot the points on the grid
    plt.scatter(x1, y1, marker='o', s=10, c='blue')  # Adjust marker and color as needed

    # Set axis limits to match the grid size
    plt.xlim(0, grid11_size)
    plt.ylim(0, grid11_size)

    # Optionally, add labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Scatter Plot on a 100x100 Grid")

    # Display the plot
    # plt.show()
    for x in range(grid_size):
        for y in range(grid_size):
            laplacian = (concentration[(x + 1) % grid_size, y] + concentration[(x - 1) % grid_size, y]
                         + concentration[x, (y + 1) % grid_size] + concentration[x, (y - 1) % grid_size]
                         - 4 * concentration[x, y]) / (dx ** 2)
            new_concentration[x, y] = concentration[x, y] + D * laplacian * dt
            
    new_concentration[chemo_source] = 1.0  # Constant chemoattractant source
    # new_concentration[(100,100)] = 1.0
    concentration = new_concentration

# Display the concentration field
plt.imshow(concentration, cmap='hot', origin='lower', extent=[0, grid_size, 0, grid_size])
plt.colorbar(label='Chemotactic Concentration')
plt.title('Keller-Segel Model Simulation')
plt.show()
