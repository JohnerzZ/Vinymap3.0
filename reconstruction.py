#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2


# ## Parameters

# In[2]:


#parameters
image_directory = "./images"
number_of_pcds = 12
downsample = True
voxel_size = 0.01
# density filtering
epsilon = 0.07
neighbour_threshold = 30
#cluster filtering
eps=0.07
min_points=6
minimum_cluster_volume = 0.015
#camera parameters
fx = 958.7215576171875
fy = 958.7215576171875
cx = 560
cy = 307.52716064453125

width = 1104
height = 621


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
    #o3d.visualization.draw_geometries([pcd])
print('pointclouds are enhanced and ready to be used' + '\n')


# ## Testing PCD & RGB Alignment
# 
# press any key to close visualization window, after it appears.

# In[5]:


# #uncomment to visualize and test pcd & rgb alignment
# #testing projection
# chosen = 2
# pcd = pcds[chosen]

# # Camera intrinsic parameters
# camera_matrix = np.array([
#     [fx, 0, cx],
#     [0, fy, cy],
#     [0, 0, 1]
# ])
# rotation_matrix = np.identity(3)
# translation_vector = np.zeros(3)

# #read image
# image_directory = "/home/johnerzz/csl_exp/src/rover_demo/nexus_4wd_mecanum_simulator_demo/nexus_4wd_mecanum_gazebo/scripts/images"
# image=cv2.imread(image_directory + f"/{chosen}.png", cv2.IMREAD_UNCHANGED)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.flip(image, 1)

# #testing cv2 point projection function
# projected_points = cv2.projectPoints(np.asarray(pcd.points), rotation_matrix, translation_vector, camera_matrix, 
#                                None)
# projected_points = projected_points[0]
# projected_points = np.array([item for sublist in projected_points for item in sublist])

# # Draw the projected points on the image
# for point in projected_points:
#     x, y = point
#     x,y = (int(x),int(y))
#     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
# # Display the image
# cv2.imshow('Projected Point Cloud', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 


# ## Mesh Creation

# In[6]:


#Create a Mesh from each pcd
ball_meshes = []
alpha_meshes = []
meshes = []
print('creating a mesh from each pcd')
print('using the BPA and alpha shapes algorithm')

for i,pcd in enumerate(pcds):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 2*avg_dist
    
    # estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))#heuristic for the radius parameter
    pcd.orient_normals_towards_camera_location()
    
    #create the mesh
    
    #print('running the ball pivoting algorithm')
    radii = list(np.arange(1.0, 2.0, 0.5))
    radii = [i * radius for i in radii]
    ball_meshes.append(o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)))
    ball_meshes[i].compute_vertex_normals()
    
    #o3d.visualization.draw_geometries([ball_meshes[i]], mesh_show_back_face=True, window_name = 'BPA Mesh')
    
    alpha = 0.02
    alpha_meshes.append(o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha))
    alpha_meshes[i].paint_uniform_color([0.3,0.3,0.3])
    alpha_meshes[i].compute_vertex_normals()
    #o3d.visualization.draw_geometries([alpha_meshes[i]], mesh_show_back_face=True, window_name = 'Alpha Mesh')
    
    meshes.append(ball_meshes[i] + alpha_meshes[i])
    #meshes.append(ball_meshes[i])
    
    #Filter the mesh

    #print("filtering")
    #meshes[i] = meshes[i].filter_smooth_simple(number_of_iterations = 1)
    #meshes[i] = meshes[i].subdivide_loop(number_of_iterations=1)
    meshes[i].compute_vertex_normals()
    print('.')
    
    #o3d.visualization.draw_geometries([meshes[i]], mesh_show_back_face=True, window_name = 'BPA Mesh')


# ## Projective Texture Mapping

# In[7]:


#Project proper RGB image to each mesh
print('projecting proper RGB images onto each mesh')
# utility function to project mesh vertices on RGB images and extract uvs
def generate_uvs2(vertices, faces, image_maxx, image_maxy, rotation_matrix, translation_vector, camera_matrix):

    projected_points = cv2.projectPoints(np.asarray(vertices), rotation_matrix, translation_vector, camera_matrix, 
                               None)
    projected_points = projected_points[0]
    uvs = []

    for face in faces:
        for i in range(3):
            uvs.append(projected_points[face[i]][0])
    
    #normalize uvs
    normalized_uvs = [[uv[0]/image_maxx, uv[1]/image_maxy] for uv in uvs]

    return normalized_uvs

# Camera intrinsic parameters
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])
rotation_matrix = np.identity(3)
translation_vector = np.zeros(3)

for i,mesh in enumerate(meshes):
    #read image
    image_directory = "./images"
    image=cv2.imread(image_directory + f"/{i}.png", cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1) 

    uvs = generate_uvs2(mesh.vertices, mesh.triangles, image.shape[1],image.shape[0], rotation_matrix,
                        translation_vector, camera_matrix)

    mesh.triangle_uvs = o3d.pybind.utility.Vector2dVector(uvs)
    mesh.textures=[o3d.geometry.Image(image)]
    mesh.triangle_material_ids = o3d.utility.IntVector([0]*len(mesh.triangles))
    print('.')
    #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, window_name = 'Enhanced mesh')


# ## Point Cloud Registration

# In[8]:


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


# In[9]:


# Register the pcds and therefore the meshes, to build the final map.
print("Registering the meshes")
max_correspondence_distance_coarse = voxel_size * 40
max_correspondence_distance_fine = voxel_size * 4
poses,bounding_boxes,full_map  = full_registration(pcds,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)


# In[10]:


# #uncomment to visualize the raw point cloud map
# full_map = full_map.voxel_down_sample(voxel_size=voxel_size)
# o3d.visualization.draw_geometries([full_map])


# ## Mesh Registration

# In[11]:


#the poses that were acquired during pcd registration are used.
for i,mesh in enumerate(meshes):
    mesh.transform(poses[i])


# In[12]:


import time

vis_mesh = meshes[0]
vis = o3d.visualization.Visualizer()
vis.create_window()
time.sleep(0.5)
for i in range(number_of_pcds):
    vis.add_geometry(vis_mesh)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.35)
    vis.get_render_option().light_on = False
    vis.poll_events()
    vis.update_renderer()
    vis.remove_geometry(vis_mesh)
    vis_mesh = vis_mesh + meshes[i]
    time.sleep(0.3)
time.sleep(5)
vis.destroy_window()


# In[13]:


print('\n' + "Visualizing enhanced mesh. Press L to toggle lighting effects." + '\n' + 'Use mouse to change view' + '\n' + 'Close window when done.')
o3d.visualization.draw_geometries([meshes[0], meshes[2], meshes[4], meshes[6], meshes[7], meshes[8],meshes[9]], mesh_show_back_face=True, window_name = 'Enhanced mesh')


# In[ ]:




