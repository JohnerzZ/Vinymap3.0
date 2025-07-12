#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import colorsys


# ## Parameters

# In[2]:


#parameters
image_directory = "./images"
number_of_pcds = 8
downsample = True
voxel_size = 0.01
# density filtering
epsilon = 0.07
neighbour_threshold = 30
#cluster filtering
eps=0.07
min_points=6
minimum_cluster_volume = 0.015


# ## Point Cloud Loading And Filtering

# In[3]:


def load_point_clouds(image_directory, number_of_pcds, downsample = True, voxel_size = 0.02, visualization = False):
    pcds = []
    for i in range(number_of_pcds):
        pcd = o3d.io.read_point_cloud(image_directory + "/%d.pcd" %(i))
        #print(i)
        # 1) Remove nfp
        pcd = pcd.remove_non_finite_points()
        # 2) Transform according to camera position
        #real-life experiment:
        transformation_matrix = np.array([[0, -1, 0, 0],
                                          [0, 0, 1, 0],
                                          [-1, 0, 0, 0],
                                          [0, 0, 0, 1]])
#         #gazebo experiment:
#         transformation_matrix = np.array([[1, 0, 0, 0],
#                                           [0, -1, 0, 0],
#                                           [0, 0, -1, 0],
#                                           [0, 0, 0, 1]])   
        pcd.transform(transformation_matrix)
        # 3) Downsample
        if downsample:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        if visualization:
            o3d.visualization.draw_geometries([pcd])
        pcds.append(pcd)
    return pcds


# 4) Limit pcd bounds to desired values using bounding boxes
# Very important step because of noisy ZED pcd
def crop_point_clouds(pcds, front, top, back, right, left, visualization = False):
    cropped_pcds = []
    #crop initial point cloud and use its cropped shape to crop every other pointcloud.
    #we need them to be as uniform as possible for registration and projection.
    initial_pcd = pcds[0]
    #first crop using a bounding_box
    aabb = initial_pcd.get_axis_aligned_bounding_box()
    corners = np.asarray(aabb.get_box_points())
    corners[[3, 4, 5, 6], 2] = front #front boundary determined here
    corners[[2, 4, 5, 7], 1] = top   #top boundary determined here
    corners[[0, 1, 2, 7], 2] = back  #back boundary determined here
    corners[[1, 4, 6,7], 0] = right  #right boundary determined here
    corners[[0, 2, 3, 5], 0] = left  #left boundary determined here
    points = o3d.utility.Vector3dVector(corners)
    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
    oriented_bounding_box.color = (0, 1, 0)
    initial_pcd = initial_pcd.crop(oriented_bounding_box)
    #then crop using a convex hull polygon and keep that polygon to crop every other pcd.
    initial_shape=create_convex_hull_polygon(initial_pcd)
    intial_pcd = initial_shape.crop_point_cloud(initial_pcd)
    cropped_pcds.append(initial_pcd)
    if visualization:
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = initial_shape.bounding_polygon
        o3d.visualization.draw_geometries([initial_pcd, oriented_bounding_box, temp_pcd])
    #crop the rest of the pcds.
    for pcd in pcds[1:]:
        pcd = initial_shape.crop_point_cloud(pcd)
        cropped_pcds.append(pcd)
        if visualization:
            o3d.visualization.draw_geometries([pcd, oriented_bounding_box, temp_pcd])
    return cropped_pcds

# utility function used in step 5
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


# 5) Remove outliers; points with too few neighbours
def density_filter_point_clouds(pcds, epsilon, neighbour_threshold, visualization = False):
    filtered_pcds = []
    for pcd in pcds:
        pcd_neighbor_counts, pcd_neighbor_ids_per_point = find_neighbors_per_point(pcd, epsilon)
        sparse_points_indexes = [] 
        for point in range(len(pcd_neighbor_counts)):
            if pcd_neighbor_counts[point] < neighbour_threshold:
                sparse_points_indexes.append(point)
        sparse_points = pcd.select_by_index(sparse_points_indexes)
        sparse_pcd = o3d.geometry.PointCloud(sparse_points)
        if visualization:
            color = np.array([1, 0, 0])  # Red
            sparse_pcd.paint_uniform_color(color)
            o3d.visualization.draw_geometries([pcd, sparse_pcd])
        dists = pcd.compute_point_cloud_distance(sparse_pcd)
        dists = np.asarray(dists)
        ind = np.where(dists > 0.05)[0]
        pcd = pcd.select_by_index(ind)
        filtered_pcds.append(pcd)
        if visualization:
            o3d.visualization.draw_geometries([pcd])
    return filtered_pcds


# 6) Remove cluster outliers; clusters that are too small.
def cluster_filter_point_clouds(pcds, eps, min_points, minimum_cluster_volume, visualization = False):
    filtered_pcds = []
    for pcd in pcds:
        labels = np.array(pcd.cluster_dbscan(eps=0.07, min_points=6, print_progress=False))
        max_label = labels.max()
        points = np.asarray(pcd.points)
        labeled_points = list(zip(labels, points))
        clusters = [[] for i in range(max_label+1)]
        # Separate the points into lists based on the cluster they belong to
        for labeled_point in labeled_points:
            if labeled_point[0] != -1:
                clusters[labeled_point[0]].append(labeled_point[1])
        # Make a pointcloud (cluster) from each of the lists
        pcd_clusters = []
        for cluster in clusters:
            pcd_cluster = o3d.geometry.PointCloud()
            pcd_cluster.points = o3d.utility.Vector3dVector(cluster)
            pcd_clusters.append(pcd_cluster)
        # Find the clusters that are too small
        pcd_filter = o3d.geometry.PointCloud()
        for pcd_cluster in pcd_clusters:
            bounding_box = pcd_cluster.get_oriented_bounding_box()
            if bounding_box.volume() < minimum_cluster_volume:
                pcd_filter += pcd_cluster
        if visualization:
            color = np.array([1, 0, 0])  # Red
            pcd_filter.paint_uniform_color(color)
            o3d.visualization.draw_geometries([pcd, pcd_filter])
        # Remove the small clusters
        dists = pcd.compute_point_cloud_distance(pcd_filter)
        dists = np.asarray(dists)
        ind = np.where(dists > 0.1)[0]
        pcd = pcd.select_by_index(ind)
        filtered_pcds.append(pcd)
        if visualization:
            o3d.visualization.draw_geometries([pcd])
    return filtered_pcds


from scipy.spatial import ConvexHull

# utility function used to cut mesh to pieces according to convex polygons for projection
def create_convex_hull_polygon(pcd):
    #create convex hull polygon
    hull_mesh = pcd.compute_convex_hull()[0]
    bounding_polygon = np.array(hull_mesh.vertices).astype("float64")
    vol = o3d.visualization.SelectionPolygonVolume()
    #volume will be extruded on Y axis (height)
    vol.orthogonal_axis = "Y"
    vol.axis_max = np.max(bounding_polygon[:, 1])
    vol.axis_min = np.min(bounding_polygon[:, 1])-0.1
    #flatten polygon to a surface on a plane defined by Y axis.
    #also retake convex hull to remove internal points
    #bounding_polygon[:, 1] = 0
    flattened_polygon = []
    for point in bounding_polygon:
        x = point[0]
        y = point[2]
        flattened_point = (x, y)
        flattened_polygon.append(flattened_point)
    #Take convex hull of flat polygon 
    vertices = ConvexHull(flattened_polygon).vertices

    # Extract x and y coordinates from the vertices list
    x_coords = [flattened_polygon[vertex][0] for vertex in vertices]
    y_coords = [flattened_polygon[vertex][1] for vertex in vertices]

    # Save the convex hull vertices in a list of elements of the form [x,y]
    final_bounding_polygon = []
    for i in range(len(vertices)):
        final_bounding_polygon.append([x_coords[i],0, y_coords[i]])

    final_bounding_polygon = np.array(final_bounding_polygon).astype("float64")
    vol.bounding_polygon = o3d.utility.Vector3dVector(final_bounding_polygon)
    return vol


# In[4]:


print('loading & preparing pointclouds')
pcds = load_point_clouds(image_directory = image_directory, number_of_pcds = number_of_pcds, 
                         downsample = downsample, voxel_size = voxel_size, visualization = False)
print('.')
pcds = crop_point_clouds(pcds = pcds, front = -0.7, top = 1.2, back = -2.3, right = 0.9, left = -1, visualization = False)
print('.')
# pcds = density_filter_point_clouds(pcds = pcds, epsilon = epsilon, 
#                                    neighbour_threshold = neighbour_threshold, visualization = True)
# pcds = cluster_filter_point_clouds(pcds = pcds, eps=eps, min_points = min_points,
#                                    minimum_cluster_volume = minimum_cluster_volume, visualization = True)
print('.')
for pcd in pcds:
    pcd.estimate_normals()
#     o3d.visualization.draw_geometries([pcd])
print('pointclouds are enhanced and ready to be used')


# ## Point Cloud Registration

# In[5]:


# ICP related functions
def initial_pairwise_registration(source, target):
    print("Applying point-to-plane ICP")
    #icp_coarse could be replaced with global registration. Keep that in mind.
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 5000))
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 5000))
    transformation_icp = icp_fine.transformation
    return transformation_icp

def pairwise_registration(source, target, transformation):
    #print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 1e-7, relative_rmse = 1e-7, max_iteration = 5000))
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 1e-7, relative_rmse = 1e-7, max_iteration = 5000))
    icp_finer = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine*0.1,
        icp_fine.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 1e-7, relative_rmse = 1e-7, max_iteration = 5000))
    icp_finest = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine*0.009,
        icp_finer.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 1e-7, relative_rmse = 1e-7, max_iteration = 5000))
    transformation_icp = icp_finest.transformation
    return transformation_icp, icp_finest.inlier_rmse


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    poses = []
    bounding_boxes = []
    local_map = o3d.geometry.PointCloud()
    global_map = o3d.geometry.PointCloud()
    running_pcd = o3d.geometry.PointCloud()
    odometry = np.identity(4)
    running_pcd = pcds[0]
    running_pcd.transform(odometry)

    bounding_boxes.append(create_convex_hull_polygon(running_pcd))
    pcd = o3d.geometry.PointCloud()
    pcd.points = bounding_boxes[0].bounding_polygon
    #o3d.visualization.draw_geometries([running_pcd, pcd])
    local_map += running_pcd
    print("pcd was added: " + str(0))
    poses.append(odometry)
    n_pcds = len(pcds)
    transformation_icp = initial_pairwise_registration(pcds[1], pcds[0])
    odometry = np.dot(transformation_icp, odometry)
    running_pcd = pcds[1]
    running_pcd.transform(odometry)
    bounding_boxes.append(create_convex_hull_polygon(running_pcd))
    pcd = o3d.geometry.PointCloud()
    pcd.points = bounding_boxes[1].bounding_polygon
    #o3d.visualization.draw_geometries([running_pcd, pcd])
    local_map += running_pcd
    print("pcd was added: " + str(1))

    global_map += local_map
    poses.append(odometry)
    
    for n_pcd in range(1, n_pcds-1):
        # we assume that two consecutive transformations are similar 
        # later consider involving the actual relative transformation between consecutive transformations
        running_pcd = pcds[n_pcd + 1]
        transformation_icp,inlier_rmse = pairwise_registration(running_pcd, local_map, 1.1*transformation_icp)
        #exclude outlier pcds that did not match accurately
        print("inlier rmse :" + str(inlier_rmse))
        odometry = transformation_icp
        running_pcd.transform(transformation_icp)
        bounding_boxes.append(create_convex_hull_polygon(running_pcd))
        pcd = o3d.geometry.PointCloud()
        pcd.points = bounding_boxes[n_pcd+1].bounding_polygon
        #o3d.visualization.draw_geometries([running_pcd, pcd])
        local_map += running_pcd
        poses.append(odometry)
        #only add good matches to the global map
        if (inlier_rmse < 0.021):
            global_map += running_pcd
            print("pcd was added: " + str(n_pcd + 1))

    return poses, bounding_boxes, global_map


# In[6]:


# Register the pcds and therefore the meshes, to build the final map.
print("Registering the point clouds")
max_correspondence_distance_coarse = voxel_size * 40
max_correspondence_distance_fine = voxel_size * 4
poses,bounding_boxes,full_map  = full_registration(pcds,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)


# In[7]:


full_map = full_map.voxel_down_sample(voxel_size=voxel_size)
# o3d.visualization.draw_geometries([full_map])


# In[8]:


# full_map = pcds[4]


# ## Separate Canopy from surroundings

# In[9]:


#crop the ground (soil and weeds)
print('')
print('separating the canopy from the rest of the environment')
fmap = o3d.geometry.PointCloud()
fmap = full_map
aabb = fmap.get_axis_aligned_bounding_box()
corners = np.asarray(aabb.get_box_points())
corners[[0, 1, 3, 6], 1] = -0.2    # ground boundary determined here
points = o3d.utility.Vector3dVector(corners)
# print(corners)
oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
oriented_bounding_box.color = (0, 1, 0)
fmap = fmap.crop(oriented_bounding_box)
# o3d.visualization.draw_geometries([fmap, oriented_bounding_box])

# Select points with green color
colors = fmap.colors
filtered_points = []
for i, point in enumerate(fmap.points):
    color = list(colorsys.rgb_to_hsv(colors[i][0],colors[i][1],colors[i][2])) 
    #extract hue, saturation and value
    H = color[0]
    S = color[1]
    V = color[2]
    # Find if pixel is green, i.e. if 100 < Hue < 180 and saturation < 180
    hlo,hhi = 85,200
    vhi = 180
    # Rescale to 0-1, rather than 0-360
    hlo = hlo / 360
    hhi = hhi / 360
    vhi = vhi / 360
    if ((H>hlo) & (H<hhi) & (V<vhi)):
        filtered_points.append(point)

canopy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(filtered_points))
canopy.paint_uniform_color([0, 1, 0])
canopy.estimate_normals()
# o3d.visualization.draw_geometries([full_map,canopy])


# ## Estimate full canopy with Alpha Shapes

# In[10]:


#create full mesh
print('estimating full canopy')
alpha = 0.11
full_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(canopy, alpha)
full_mesh.paint_uniform_color([0.3,0.3,0.3])
full_mesh = full_mesh.filter_smooth_simple(number_of_iterations = 2)
full_mesh.compute_vertex_normals()

# o3d.visualization.draw_geometries([full_map,full_mesh], mesh_show_back_face=True, window_name = 'BPA Mesh')


# In[11]:


full_mesh_cloud = full_mesh.sample_points_uniformly(20000)
full_mesh_cloud.paint_uniform_color([0, 1, 0])
# o3d.visualization.draw_geometries([full_map, full_mesh_cloud, canopy], mesh_show_back_face=True, window_name = 'BPA Mesh')


# In[12]:


print('finding holes')
#compute distances of all points to the green canopy
dists = full_mesh_cloud.compute_point_cloud_distance(fmap)
dists = np.asarray(dists)
#remove all the points that are on or close to the green canopy
ind = np.where(dists > 0.035)[0]
holes = o3d.geometry.PointCloud()
holes = full_mesh_cloud.select_by_index(ind)
holes.paint_uniform_color([1, 0, 0])
# o3d.visualization.draw_geometries([holes], mesh_show_back_face=True, window_name = 'BPA Mesh')


# In[13]:


o3d.visualization.draw_geometries([full_map, holes], mesh_show_back_face=True, window_name = 'BPA Mesh')


# ## Calculate Density Index

# In[14]:


print('calculating canopy density index')
print('')
holes = holes.voxel_down_sample(voxel_size=0.02)
full_mesh_cloud = full_mesh_cloud.voxel_down_sample(voxel_size=0.02)
# o3d.visualization.draw_geometries([holes], mesh_show_back_face=True, window_name = 'BPA Mesh')
# o3d.visualization.draw_geometries([full_mesh_cloud], mesh_show_back_face=True, window_name = 'BPA Mesh')

hole_points = len(holes.points)
full_points = len(full_mesh_cloud.points)

ratio = hole_points / full_points
print('CDI is: ' + str(ratio))


# In[ ]:




