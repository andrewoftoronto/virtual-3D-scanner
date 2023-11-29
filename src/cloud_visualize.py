import numpy as np

def visualize(point_cloud, colours):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

    keeps = np.linalg.norm(point_cloud, axis=1) < 30
    point_cloud = point_cloud[keeps]
    colours = colours[keeps]

    # Ensure point_cloud has a sane number of samples
    keeps = np.random.randint(len(point_cloud), size=10**4)
    point_cloud = point_cloud[keeps]
    colours = colours[keeps]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=colours)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()
    print("Plotted")