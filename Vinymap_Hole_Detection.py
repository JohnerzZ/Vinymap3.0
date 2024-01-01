#!/usr/bin/env python
# coding: utf-8

# In[1]:


import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

image_directory = "/home/johnerzz/csl_exp/src/rover_demo/nexus_4wd_mecanum_simulator_demo/nexus_4wd_mecanum_gazebo/scripts/images"

pcd = o3d.io.read_point_cloud(image_directory + "/10(old).pcd")

pcd = pcd.remove_non_finite_points()
print(pcd)

pcd.transform([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
pcd.transform([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])


# In[40]:


# pick regular points and make seed pointclouds
initial_points = np.random.uniform(np.subtract(pcd.get_center(),[2,1,1]) , np.add(pcd.get_center(),[2,1,1]), size=(100000, 3))


# In[41]:


seeds = o3d.geometry.PointCloud()
seeds.points = o3d.utility.Vector3dVector(initial_points)
seeds = seeds.voxel_down_sample(voxel_size=0.5)


# In[42]:


o3d.visualization.draw_geometries([seeds,pcd])


# In[43]:


# define all space
points = np.random.uniform(np.subtract(pcd.get_center(),[2,1,1]) , np.add(pcd.get_center(),[2,1,1]), size=(30000, 3))
whole_space = o3d.geometry.PointCloud()
whole_space.points = o3d.utility.Vector3dVector(points)
dilation_radius =  0.1
seed_distance = 0.15

#dilate and clean
for i in range(4):
    # dilate in free space
    dists = whole_space.compute_point_cloud_distance(seeds)
    dists = np.asarray(dists)
    ind = np.where(dists < dilation_radius)[0]
    dilated_pcd = whole_space.select_by_index(ind)
    seeds+=dilated_pcd
    # clean
    dists = seeds.compute_point_cloud_distance(pcd)
    dists = np.asarray(dists)
    ind= np.where(dists > seed_distance)[0]
    seeds = seeds.select_by_index(ind)


# In[44]:


o3d.visualization.draw_geometries([seeds,pcd])


# In[45]:


################################ EROSION. Consider Running this more than once ################################
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


# In[46]:


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud = cloud.select_by_index(ind)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    


# In[47]:



# Set the epsilon radius

epsilon = 0.1


# Find the number of neighbors for each point
pcd_neighbor_counts, pcd_neighbor_ids_per_point = find_neighbors_per_point(seeds, epsilon)

#color points with less neighbors as red
#save indexes of those points to create a new pointcloud out of them later
sparse_points_indexes = [] 
for i in range(len(pcd_neighbor_counts)):
    if pcd_neighbor_counts[i] < 6:
    #if pcd_neighbor_counts[i] < 0.03*pcd_mean_density:
        sparse_points_indexes.append(i)


# In[48]:


display_inlier_outlier(seeds,sparse_points_indexes)


# In[49]:


#Remove points with too few neighbours
sparse_points = seeds.select_by_index(sparse_points_indexes)

# Create a new point cloud from the extracted points
sparse_pcd = o3d.geometry.PointCloud(sparse_points)

#compute distances of all points to the sparse pointcloud
dists = seeds.compute_point_cloud_distance(sparse_pcd)
dists = np.asarray(dists)
#remove all the sparse points as well as points very near them
ind = np.where(dists > 0.1)[0]
seeds = seeds.select_by_index(ind)
# Visualize the new point cloud
o3d.visualization.draw_geometries([seeds])

################################ EROSION ends here################################


# In[50]:


# cluster

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        seeds.cluster_dbscan(eps=0.1, min_points=8, print_progress=True)) #0.032 5

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
seeds.colors = o3d.utility.Vector3dVector(colors[:, :3])


# In[51]:


o3d.visualization.draw_geometries([seeds,pcd])


# In[52]:


#make separate pointclouds from each cluster
points = np.asarray(seeds.points)
labeled_points = list(zip(labels, points))
clusters = [[] for i in range(max_label+1)]
for labeled_point in labeled_points:
    if labeled_point[0] != -1:
        clusters[labeled_point[0]].append(labeled_point[1])
pcd_clusters = []
for cluster in clusters:
    pcd_cluster = o3d.geometry.PointCloud()
    pcd_cluster.points = o3d.utility.Vector3dVector(cluster)
    pcd_clusters.append(pcd_cluster)


# In[53]:


#Remove outliers
biggest_cluster = 0
largest_volume = 0

#remove largest cluster
for i, pcd_cluster in enumerate(pcd_clusters):
    if len(pcd_cluster.points) > 8:
        bounding_box = pcd_cluster.get_oriented_bounding_box()
        if bounding_box.volume() > largest_volume:
            biggest_cluster = i
            largest_volume = bounding_box.volume()
print(biggest_cluster)
holes = o3d.geometry.PointCloud()
for i, pcd_cluster in enumerate(pcd_clusters):
    if i != biggest_cluster:
        holes += pcd_cluster


# In[54]:


o3d.visualization.draw_geometries([holes,pcd])


# In[55]:


#remove outlier clusters
dists = holes.compute_point_cloud_distance(pcd)
dists = np.asarray(dists)
ind= np.where(dists < 0.25)[0]
holes = holes.select_by_index(ind)


# In[56]:


o3d.visualization.draw_geometries([holes,pcd])


# In[ ]:




