#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2


# In[2]:


#parameters
image_directory = "/home/johnerzz/csl_exp/src/rover_demo/nexus_4wd_mecanum_simulator_demo/nexus_4wd_mecanum_gazebo/scripts/images"
number_of_pcds = 10
downsample = True
voxel_size = 0.02
# density filtering
epsilon = 0.07
neighbour_threshold = 30
#cluster filtering
eps=0.07
min_points=6
minimum_cluster_volume = 0.015
#camera parameters
fx = 516.127
fy = 516.127
cx = 539.95
cy = 313.714


# In[3]:


def load_point_clouds(image_directory, number_of_pcds, downsample = True, voxel_size = 0.02, visualization = False):
    pcds = []
    for i in range(number_of_pcds):
        pcd = o3d.io.read_point_cloud(image_directory + "/%d.pcd" %(i+1))
        # 1) Remove nfp
        pcd = pcd.remove_non_finite_points()
        # 2) Transform according to camera position
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
def crop_point_clouds(pcds, front, back, right, left, visualization = False):
    cropped_pcds = []
    for pcd in pcds:
        aabb = pcd.get_axis_aligned_bounding_box()
        corners = np.asarray(aabb.get_box_points())
        corners[[2, 4, 5, 7], 1] = front #front boundary determined here
        corners[[0, 1, 2, 7], 2] = back  #back boundary determined here
        corners[[1, 4, 6,7], 0] = right  #right boundary determined here
        corners[[0, 2, 3, 5], 0] = left  #left boundary determined here
        points = o3d.utility.Vector3dVector(corners)
        oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points)
        oriented_bounding_box.color = (0, 1, 0)
        pcd = pcd.crop(oriented_bounding_box)
        cropped_pcds.append(pcd)
        if visualization:
            o3d.visualization.draw_geometries([pcd, oriented_bounding_box])
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
    vol.axis_min = np.min(bounding_polygon[:, 1])
    #flatten polygon to a surface on a plane defined by Y axis.
    #also retake convex hull to remove internal points
    #bounding_polygon[:, 1] = 0
    flattened_polygon = []
    for point in bounding_polygon:
        x = point[0]
        z = point[2]
        flattened_point = (x, z)
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


# In[4]:


# ICP related functions
def initial_pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
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
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 1e-7, relative_rmse = 1e-7, max_iteration = 5000))
    icp_finer = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine*0.1,
        icp_fine.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 1e-7, relative_rmse = 1e-7, max_iteration = 5000))
    icp_finest = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine*0.02,
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
    local_map += running_pcd
    poses.append(odometry)
    n_pcds = len(pcds)
    transformation_icp = initial_pairwise_registration(pcds[1], pcds[0])
    odometry = np.dot(transformation_icp, odometry)
    running_pcd = pcds[1]
    running_pcd.transform(odometry)
    bounding_boxes.append(create_convex_hull_polygon(running_pcd))
    local_map += running_pcd

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
        local_map += running_pcd
        poses.append(odometry)
        #only add good matches to the global map
        if (inlier_rmse < 0.021):
            global_map += running_pcd
            print("pcd was added: " + str(n_pcd))

    return poses, bounding_boxes, global_map


# In[5]:


print('preparing pointclouds')
pcds = load_point_clouds(image_directory = image_directory, number_of_pcds = number_of_pcds, 
                         downsample = downsample, voxel_size = voxel_size, visualization = False)
print('.')
pcds = crop_point_clouds(pcds = pcds, front = 1.4, back = -2.2, right = 2.0, left = -1.2, visualization = False)
print('.')
# pcds = density_filter_point_clouds(pcds = pcds, epsilon = epsilon, 
#                                    neighbour_threshold = neighbour_threshold, visualization = True)
pcds = cluster_filter_point_clouds(pcds = pcds, eps=eps, min_points = min_points,
                                   minimum_cluster_volume = minimum_cluster_volume, visualization = False)
print('.')
for pcd in pcds:
    pcd.estimate_normals()
    #o3d.visualization.draw_geometries([pcd])
print('pointclouds are clean and ready to be used')


# In[6]:


print("Full registration ...")
max_correspondence_distance_coarse = voxel_size * 40
max_correspondence_distance_fine = voxel_size * 4
poses,bounding_boxes,full_map  = full_registration(pcds,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)


# In[7]:


o3d.visualization.draw_geometries([full_map])
full_map = full_map.voxel_down_sample(voxel_size=voxel_size)
#o3d.visualization.draw_geometries([full_map])


# In[58]:


#Create a Mesh from the full_map

distances = full_map.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 2*avg_dist
    
# estimate normals
full_map.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))#heuristic for the radius parameter
full_map.orient_normals_towards_camera_location()
    
#create the mesh
    
print('running the ball pivoting algorithm')
radii = list(np.arange(1.0, 2.0, 0.5))
radii = [i * radius for i in radii]
ball_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        full_map, o3d.utility.DoubleVector(radii))
ball_mesh = ball_mesh.filter_smooth_simple(number_of_iterations = 1)
ball_mesh.compute_vertex_normals()
    
o3d.visualization.draw_geometries([ball_mesh], mesh_show_back_face=True, window_name = 'BPA Mesh')

print('running the alpha algorithm')
alpha = 0.03
alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(full_map, alpha)
alpha_mesh = alpha_mesh.filter_smooth_simple(number_of_iterations = 1)
alpha_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([alpha_mesh], mesh_show_back_face=True, window_name = 'Alpha Mesh')
    
mesh = ball_mesh + alpha_mesh
mesh.paint_uniform_color([0.3,0.3,0.3])
#Filter the mesh

print("filtering")
mesh = mesh.merge_close_vertices(eps=0.01)
mesh = mesh.filter_smooth_simple(number_of_iterations = 1)
mesh.compute_vertex_normals()
    
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, window_name = 'BPA Mesh')


# In[35]:


#projecting using raycasting

# Camera intrinsic parameters
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

meshes = []
print("projecting the images on the whole mesh")
for i in range(10):
    running_mesh = bounding_boxes[i].crop_triangle_mesh(mesh)
    #o3d.visualization.draw_geometries([running_mesh], mesh_show_back_face=False, window_name = 'Enhanced mesh')



    cam_rotation_matrix = np.array([[np.cos(np.pi), 0, np.sin(np.pi), 0],
                                 [0, 1, 0, 0],
                                 [-np.sin(np.pi), 0, np.cos(np.pi), 0],
                                 [0, 0, 0, 1]])

    #change mesh format
    running_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(running_mesh)

    # Descaling
    scale_factor = poses[i][3][3]
    descale = 1/scale_factor
    C = np.array([[descale, 0, 0, 0],
                  [0, descale, 0, 0],
                  [0, 0, descale, 0],
                  [0, 0, 0, descale]])

    new_pose = C@poses[i]

    #Get matrices
    rotation_matrix = new_pose[:3, :3]
    rotation_matrix = rotation_matrix.T
    translation_vector = new_pose[:3, 3]

    # Create the transformation matrix
    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation_vector
    transformation_matrix[3,3] = 1

    #perform raycasting
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(running_mesh_t)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(intrinsic_matrix = camera_matrix, 
                                                              extrinsic_matrix = np.matmul(transformation_matrix, cam_rotation_matrix),
                                                              width_px = 1104,
                                                              height_px = 621)

    ans = scene.cast_rays(rays)
#     #raycasting visualization
#     hit = ans['t_hit'].isfinite()
#     points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
#     pcd = o3d.t.geometry.PointCloud(points)
#     o3d.visualization.draw_geometries([pcd.to_legacy()])
    # the triangles that got hit by the rays are the visible triangles
    visible_triangles = list(set(ans['primitive_ids'].numpy().flatten()))
    
    #keep only the visible triangles
    to_remove_triangles = [1]*(len(running_mesh.triangles))
    for triangle_index in visible_triangles[:-1]:
        #do not remove the visible triangles
        to_remove_triangles[triangle_index] = 0
    running_mesh.remove_triangles_by_mask(to_remove_triangles)
    running_mesh.remove_unreferenced_vertices()

    #o3d.visualization.draw_geometries([running_mesh], mesh_show_back_face=True, window_name = 'Enhanced mesh')


    #project on the visible triangles
    #Get matrices
    rotation_matrix = new_pose[:3, :3]
    rotation_matrix = rotation_matrix.T
    translation_vector = -new_pose[:3, 3]
    if i != 6:
    #Read image
        image_directory = "/home/johnerzz/csl_exp/src/rover_demo/nexus_4wd_mecanum_simulator_demo/nexus_4wd_mecanum_gazebo/scripts/images"
        image=cv2.imread(image_directory + f"/{i}.png", cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        #Generate uvs and apply textures
        uvs = generate_uvs2(running_mesh.vertices, running_mesh.triangles, image.shape[1],image.shape[0], rotation_matrix,
                        translation_vector, camera_matrix)
        running_mesh.triangle_uvs = o3d.pybind.utility.Vector2dVector(uvs)
        running_mesh.textures=[o3d.geometry.Image(image)]
        running_mesh.triangle_material_ids = o3d.utility.IntVector([0]*len(running_mesh.triangles))
        o3d.visualization.draw_geometries([running_mesh], mesh_show_back_face=True, window_name = 'Enhanced mesh')
    meshes.append(running_mesh)
print("done")


# In[83]:


o3d.visualization.draw_geometries([meshes[8],meshes[7], meshes[5], meshes[4], meshes[3], meshes[2], meshes[1], meshes[0]], mesh_show_back_face=True, window_name = 'Enhanced mesh')


# In[36]:


new_mesh = meshes[0] + meshes[1] + meshes[2] + meshes[3] + meshes[4] + meshes[5] + meshes[7] + meshes[8]


# In[37]:


o3d.visualization.draw_geometries([new_mesh], mesh_show_back_face=True, window_name = 'Enhanced mesh')


# In[ ]:




