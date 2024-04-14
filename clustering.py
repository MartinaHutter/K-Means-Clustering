import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # Importing this module for 3D projection

# Load data from Excel
df = pd.read_excel('test2.xlsx')

# Assuming the Excel file has six columns containing the coordinates
points_data = df.iloc[:, :6].values

# Define the number of clusters and iteration
clusters_n = 5
iteration_n = 100

# Convert the numpy array to a TensorFlow constant
points = tf.constant(points_data, dtype=tf.float32)

# Initialize centroids
centroids = tf.Variable(tf.slice(tf.random.shuffle(points), [0, 0], [clusters_n, -1]))

# Expand dimensions for broadcasting
points_expanded = tf.expand_dims(points, 0)
centroids_expanded = tf.expand_dims(centroids, 1)

# Compute distances and assignments
distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
assignments = tf.argmin(distances, 0)

# Compute new centroids
means = []
for c in range(clusters_n):
    means.append(tf.reduce_mean(
      tf.gather(points, 
                tf.reshape(
                  tf.where(
                    tf.equal(assignments, c)
                  ),[1,-1])
               ),axis=1))
new_centroids = tf.concat(means, axis=0)

# Update centroids operation
update_centroids = centroids.assign(new_centroids)

# Initialize variables
init = tf.compat.v1.global_variables_initializer()

# Disable eager execution
tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as sess:
    # Initialize global variables
    sess.run(init)
    
    # Run the optimization loop
    for step in range(iteration_n):
        _, centroid_values, points_values, assignment_values = sess.run([update_centroids, centroids, points, assignments])
    
    # Print the centroid values
    print("centroids", centroid_values)
    print("assignments", assignment_values)

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points_values[:, 2], points_values[:, 3], points_values[:, 4], c=assignment_values, s=50, alpha=0.5)
    ax.scatter(centroid_values[:, 2], centroid_values[:, 3], centroid_values[:, 4], marker='x', color='red', label='Centroids', s=100)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.legend()
    plt.show()

# New point
new_point = [1, 2000, 10, 45, 79, 0.2]

# Compute squared distances between each centroid and the new point
squared_distances = np.sum((centroid_values[:, 2:] - new_point[2:]) ** 2, axis=1)

# Find the index of the centroid with the minimum distance (closest cluster)
closest_cluster = np.argmin(squared_distances)

print("New point:", new_point)
print("Belongs to cluster:", closest_cluster)
