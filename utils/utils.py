import streamlit as st
from pandaset import geometry
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np
import random
import folium
import open3d as o3d


def depth_projection(seq, lidar, camera_name, seq_idx):
    points3d_lidar_xyz = lidar.data[seq_idx].to_numpy()[:, :3]
    choosen_camera = seq.camera[camera_name]
    projected_points2d, camera_points_3d, inner_indices = geometry.projection(
        lidar_points=points3d_lidar_xyz,
        camera_data=choosen_camera[seq_idx],
        camera_pose=choosen_camera.poses[seq_idx],
        camera_intrinsics=choosen_camera.intrinsics,
        filter_outliers=True
    )

    # Image before projection
    ori_image = seq.camera[camera_name][seq_idx]
    st.image(ori_image, caption='Original Image', use_column_width=True)

    # Image after projection
    distances = np.sqrt(np.sum(np.square(camera_points_3d), axis=-1))
    colors = cm.jet(distances / np.max(distances))

    fig, ax = plt.subplots()
    ax.imshow(ori_image)
    ax.scatter(projected_points2d[:, 0], projected_points2d[:, 1], color=colors, s=1)
    ax.axis('off')
    st.pyplot(fig)

def semantic_segmentation(seq, lidar, camera_name, seq_idx):
    points3d_lidar_xyz = lidar.data[seq_idx].to_numpy()[:, :3]
    choosen_camera = seq.camera[camera_name]
    projected_points2d, camera_points_3d, inner_indices = geometry.projection(
        lidar_points=points3d_lidar_xyz,
        camera_data=choosen_camera[seq_idx],
        camera_pose=choosen_camera.poses[seq_idx],
        camera_intrinsics=choosen_camera.intrinsics,
        filter_outliers=True
    )
    # Image before projection
    ori_image = seq.camera[camera_name][seq_idx]
    st.image(ori_image, caption='Original Image', use_column_width=True)

    # Load semantic segmentation
    semseg = seq.semseg[seq_idx].to_numpy()

    # Get semantic segmentation on image by filtering outside points
    semseg_on_image = semseg[inner_indices].flatten()

    # Randomly generate colors for semantic segmentation
    max_seg_id = np.max(semseg_on_image)
    color_maps = [(random.random(), random.random(), random.random()) for _ in range(max_seg_id + 1)]
    colors = np.array([color_maps[seg_id] for seg_id in semseg_on_image])

    fig, ax = plt.subplots()
    ax.imshow(ori_image)
    ax.scatter(projected_points2d[:, 0], projected_points2d[:, 1], color=colors, s=1)
    ax.axis('off')  # Remove axis labels and marks
    st.pyplot(fig)

def display_map(seq):
    seq.load_gps()
    
    lats = [x['lat'] for x in seq.gps]
    longs = [x['long'] for x in seq.gps]
    
    mean_lat = lats[len(lats)//2]
    mean_long = longs[len(longs)//2]
    
    # Create a folium map centered around the mean lat/long
    folium_map = folium.Map(location=[mean_lat, mean_long], zoom_start=18)
    
    # Add the GPS points to the map
    folium.PolyLine(list(zip(lats, longs)), color='cornflowerblue', weight=10).add_to(folium_map)
    
    # Save the map to an HTML file
    folium_map.save(f'map.html')
    
    # Read the HTML file and display it in Streamlit
    with open(f'map.html', 'r') as f:
        html_string = f.read()
    
    st.components.v1.html(html_string, width=700, height=500)

def lidar3d(seq,seq_idx):
    # get Pandar64 points
    seq.lidar.set_sensor(0)
    pandar64_points = seq.lidar[seq_idx].to_numpy()
    print("Pandar64 has points: ", pandar64_points.shape)

    # get PandarGT points
    seq.lidar.set_sensor(1)
    pandarGT_points = seq.lidar[seq_idx].to_numpy()
    print("PandarGT has points: ", pandarGT_points.shape)

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

    p64_pc = o3d.geometry.PointCloud()
    p64_pc.points = o3d.utility.Vector3dVector(pandar64_points[:, :3])
    p64_pc.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(pandar64_points.shape[0])])

    gt_pc = o3d.geometry.PointCloud()
    gt_pc.points = o3d.utility.Vector3dVector(pandarGT_points[:, :3])
    gt_pc.colors = o3d.utility.Vector3dVector([[10, 0, 1] for _ in range(pandarGT_points.shape[0])])

    o3d.visualization.draw_geometries([axis_pcd, p64_pc, gt_pc], window_name="world frame")

