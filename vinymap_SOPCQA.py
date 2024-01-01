#!/usr/bin/env python
# coding: utf-8

# In[41]:


import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np


# In[42]:


def find_neighbors_per_point(pointcloud, epsilon):
    # Convert pointcloud to numpy array
    points = np.asarray(pointcloud.points)

    # Create a KDTree for efficient nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(pointcloud)

    # Initialize an empty array to store neighbor counts
    neighbor_counts = np.zeros(len(points))
    neighbor_ids_per_point = []

    # Loop through each point in the point cloud
    for i, point in enumerate(points):
        # Find all neighbors within a radius of epsilon
        [number_of_neighbours, ids, _] = kdtree.search_radius_vector_3d(point, epsilon)

        # Count the number of neighbors
        neighbor_counts[i] = number_of_neighbours
        neighbor_ids_per_point.append(ids)

    return neighbor_counts, neighbor_ids_per_point


# In[43]:


image_directory = "/home/johnerzz/csl_exp/src/rover_demo/nexus_4wd_mecanum_simulator_demo/nexus_4wd_mecanum_gazebo/scripts/images"

# examples: 2.pcd 10.pcd
pcd = o3d.io.read_point_cloud(image_directory + "/30.pcd")

pcd = pcd.remove_non_finite_points()
print(pcd)

#pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd.transform([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
pcd.transform([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd])


# In[27]:


#Later the index voxelization method can be used to make it real time. 
#Because we will calculate the density for less points
#Thus speeding up the process significantly
#used this query to find it: can you voxelize a pointcloud and get the index of the most central point of the pointcloud within each voxel using open3d python?


# In[28]:


# Set the epsilon radius
epsilon = 0.05

# Find the number of neighbors for each point
pcd_neighbor_counts, pcd_neighbor_ids_per_point = find_neighbors_per_point(pcd, epsilon)

pcd_mean_density = sum(pcd_neighbor_counts) / len(pcd_neighbor_counts) 
# Print the pointcloud mean neighbors per point
print(int(pcd_mean_density))

#This density is not exactly what we need to use as a threshold. 
#Very dense areas of the pointcloud raise this density a lot 
#Because there are too many points in those areas, thus elevating the mean
#it is not representative of a density which we would consider sparse.
#Instead we need to voxelize the pointcloud and get the density of a point within each voxel
#I will do it heuristically for now.


# In[29]:


#color points with less neighbors as red
#save indexes of those points to create a new pointcloud out of them later
sparse_points_indexes = [] 
for i in range(len(pcd_neighbor_counts)):
    if pcd_neighbor_counts[i] < 30:
    #if pcd_neighbor_counts[i] < 0.03*pcd_mean_density:
        sparse_points_indexes.append(i)
        np.asarray(pcd.colors)[i, :] = [1, 0, 0]


# In[30]:


o3d.visualization.draw_geometries([pcd])


# In[8]:


# Extract the sparse points
sparse_points = pcd.select_by_index(sparse_points_indexes)

# Create a new point cloud from the extracted points
sparse_pcd = o3d.geometry.PointCloud(sparse_points)

# Visualize the new point cloud
o3d.visualization.draw_geometries([sparse_pcd])


# In[9]:


# Convert the sparse point cloud to a mesh
print('running the ball pivoting algorithm')
#estimating normals
sparse_pcd = sparse_pcd.voxel_down_sample(voxel_size=0.01)
sparse_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))#heuristic for the radius parameter
sparse_pcd.orient_normals_to_align_with_direction([0,1,1])

radii = [2*epsilon, 4*epsilon]
sparse_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    sparse_pcd, o3d.utility.DoubleVector(radii))
sparse_mesh.paint_uniform_color([0.3,0.3,0.3])
o3d.visualization.draw_geometries([sparse_mesh], mesh_show_back_face=False, window_name = 'Sparse Mesh')

# Calculate the area of the mesh
sparse_area = sparse_mesh.get_surface_area()
print(sparse_area)


# In[10]:


# Convert the whole point cloud to a mesh
print('running the ball pivoting algorithm')
#estimating normals
down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))#heuristic for the radius parameter
down_pcd.orient_normals_to_align_with_direction([0,1,1])


# In[11]:


radii = [epsilon, 2*epsilon]
whole_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    down_pcd, o3d.utility.DoubleVector(radii))
whole_mesh.paint_uniform_color([0.3,0.3,0.3])
o3d.visualization.draw_geometries([whole_mesh], mesh_show_back_face=False, window_name = 'Initial Mesh')

# Calculate the area of the mesh
whole_area = whole_mesh.get_surface_area()
print(whole_area)


# In[91]:


############## Decide if refinement should be applied or not ##################


# In[12]:


sparsity_index = sparse_area/whole_area
print("sparsity index = " + str(sparsity_index))
if sparsity_index > 0.4:
    print("Refinement is neccessary")
else:
    print("Refinement is not neccessary")


# In[81]:


with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps = 0.03, min_points=30,print_progress=True))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])


# In[ ]:


############################################## TESTING #######################################################


# In[173]:


import open3d as o3d
import numpy as np

image_directory = "/home/johnerzz/csl_exp/src/rover_demo/nexus_4wd_mecanum_simulator_demo/nexus_4wd_mecanum_gazebo/scripts/images"

# examples: 2.pcd 10.pcd
pcd = o3d.io.read_point_cloud(image_directory + "/2.pcd")

pcd = pcd.remove_non_finite_points()
print(pcd)

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])


# In[174]:


################ creating a low density area ###############


# In[175]:


print("Testing kdtree in Open3D...")
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

print("Paint the 1500th point red.")
pcd.colors[100000] = [1, 0, 0]

pcd.colors[115000] = [1, 0, 0]


# In[176]:


print("Find its neighbors with distance less than 0.2, and paint them green.")
[k, idx1, _] = pcd_tree.search_radius_vector_3d(pcd.points[100000], 0.2)
np.asarray(pcd.colors)[idx1[1:], :] = [0, 1, 0]

[k, idx2, _] = pcd_tree.search_radius_vector_3d(pcd.points[115000], 0.2)
np.asarray(pcd.colors)[idx2[1:], :] = [0, 1, 0]


# In[177]:


print("Visualize the point cloud.")
o3d.visualization.draw_geometries([pcd])


# In[178]:


# Select the points from the original pointcloud corresponding to the list of indices
# combine the two holes into one list of points
idx = list(idx1) + list(set(idx2) - set(idx1))
selected_points = pcd.select_by_index(idx) 
# Remove points from pcd
dists = pcd.compute_point_cloud_distance(selected_points)
dists = np.asarray(dists)
ind = np.where(dists > 0.001)[0]
pcd = pcd.select_by_index(ind)
pcd.paint_uniform_color([0.5, 0.5, 0.5])
#o3d.visualization.draw_geometries([pcd])

# Create a new pointcloud from the selected points
hole_pcd = o3d.geometry.PointCloud()
hole_pcd.points = selected_points.points
hole_pcd.paint_uniform_color([0.5, 0.5, 0.5])
hole_pcd = hole_pcd.voxel_down_sample(voxel_size=0.01)
#o3d.visualization.draw_geometries([hole_pcd])


# In[179]:


pcd += hole_pcd
o3d.visualization.draw_geometries([pcd])


# In[180]:


# Set the epsilon radius
epsilon = 0.02
# Set the sensitivity (minimum required neighbours)
k = 20

# Find the number of neighbors for each point
pcd_neighbor_counts, pcd_neighbor_ids_per_point = find_neighbors_per_point(pcd, epsilon)

pcd_mean_density = sum(pcd_neighbor_counts) / len(pcd_neighbor_counts) 
# Print the pointcloud mean neighbors per point
print(int(pcd_mean_density))

#color points with less neighbors as red
#save indexes of those points to create a new pointcloud out of them later
sparse_points_indexes = [] 
for i in range(len(pcd_neighbor_counts)):
    if pcd_neighbor_counts[i] < k:
    #if pcd_neighbor_counts[i] < 0.03*pcd_mean_density:
        sparse_points_indexes.append(i)
        np.asarray(pcd.colors)[i, :] = [1, 0, 0]
        
o3d.visualization.draw_geometries([pcd])


# In[181]:


k = 13
pcd.paint_uniform_color([0.5, 0.5, 0.5])

#save indexes of those points to create a new pointcloud out of them later
sparse_points_indexes = [] 
for i in range(len(pcd_neighbor_counts)):
    if pcd_neighbor_counts[i] < k:
    #if pcd_neighbor_counts[i] < 0.03*pcd_mean_density:
        sparse_points_indexes.append(i)
        np.asarray(pcd.colors)[i, :] = [1, 0, 0]
        
o3d.visualization.draw_geometries([pcd])


# In[182]:


k = 10
pcd.paint_uniform_color([0.5, 0.5, 0.5])

#save indexes of those points to create a new pointcloud out of them later
sparse_points_indexes = [] 
for i in range(len(pcd_neighbor_counts)):
    if pcd_neighbor_counts[i] < k:
    #if pcd_neighbor_counts[i] < 0.03*pcd_mean_density:
        sparse_points_indexes.append(i)
        np.asarray(pcd.colors)[i, :] = [1, 0, 0]
        
o3d.visualization.draw_geometries([pcd])


# In[35]:


pcd = o3d.io.read_point_cloud(image_directory + "/5.pcd")
pcd = pcd.remove_non_finite_points()
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.04, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])


# In[ ]:




