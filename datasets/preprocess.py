import numpy as np
from datasets.load_labels import read_label_file
from datasets.load_calibration import Calibration
import os
import open3d as o3d

def is_point_in_box3d(point, box_center, box_dims, angle):
    """
    Check if a point lies within a rotated 3D bounding box.
    """
    translated = point - box_center

    # Rotate point back to box-aligned frame
    cosa = np.cos(-angle)
    sina = np.sin(-angle)
    rot_mat = np.array([
        [cosa, 0, -sina],
        [0,    1,  0   ],
        [sina, 0,  cosa]
    ])
    local_point = np.dot(rot_mat, translated)

    h, w, l = box_dims
    in_x = np.abs(local_point[0]) <= l / 2
    in_y = np.abs(local_point[1]) <= h / 2
    in_z = np.abs(local_point[2]) <= w / 2

    return in_x and in_y and in_z

def rotate_pc_along_y(pc, angle):
    """ Rotate point cloud along Y-axis (up). """
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    rot_mat = np.array([
        [cos_val, 0, sin_val],
        [0, 1, 0],
        [-sin_val, 0, cos_val]
    ])
    pc[:, 0:3] = pc[:, 0:3] @ rot_mat.T
    return pc

def extract_frustum_data(lidar_points, labels, calib, type_filter='Car'):
    data = []
    for obj in labels:
        if obj['type'] != type_filter:
            continue

        box2d = obj['bbox']  # x1, y1, x2, y2
        pc_in_box2d = calib.get_frustum(box2d, lidar_points)

        if pc_in_box2d.shape[0] == 0:
            continue

        # Transform to camera coordinates
        pc_cam = calib.project_velo_to_cam(pc_in_box2d)

        obj_center = np.array(obj['location'])
        angle = obj['rotation_y']
        dims = np.array(obj['dimensions'])  # h, w, l

        # Segmentation labels: 1 for points inside box, 0 outside
        seg_label = np.zeros((pc_cam.shape[0],), dtype=np.int64)
        for i in range(pc_cam.shape[0]):
            if is_point_in_box3d(pc_cam[i], obj_center, dims, angle):
                seg_label[i] = 1

        # Translate and rotate for alignment
        shifted_pc = pc_cam - obj_center
        aligned_pc = rotate_pc_along_y(shifted_pc, -angle)
        # Debugging the unique segmentation values
        #unique_vals = np.unique(seg_label)
        #print("Seg label unique values:", unique_vals)
        data.append({
            'point_cloud': aligned_pc.astype(np.float32),
            'seg_label': seg_label,
            'box_center': obj_center.astype(np.float32),
            'box_angle': angle,
            'box_dims': dims.astype(np.float32),
            'class': obj['type']
        })

    return data

def visualize_point_cloud_with_bboxes(point_cloud, seg_label, box_center, box_dims, box_angle):
    """
    Visualize point cloud with bounding boxes.
    """
    # Separate foreground and background points based on segmentation labels
    foreground_points = point_cloud[seg_label == 1]
    background_points = point_cloud[seg_label == 0]

    # Create Open3D point cloud objects
    fg_pcd = o3d.geometry.PointCloud()
    fg_pcd.points = o3d.utility.Vector3dVector(foreground_points)
    fg_pcd.paint_uniform_color([1, 0, 0])  # Red for foreground points
    
    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(background_points)
    bg_pcd.paint_uniform_color([0, 0, 1])  # Blue for background points

    # Create a bounding box from the box center, dimensions, and rotation angle
    # Create a bounding box mesh
    #box = create_3d_box(box_center, box_dims, box_angle)
    
    # Visualize the point clouds and the bounding box
    #o3d.visualization.draw_geometries([fg_pcd, bg_pcd, box])

def create_3d_box(center, dims, angle):
    """
    Create a 3D bounding box mesh given center, dimensions, and rotation angle.
    """
    h, w, l = dims
    # Create 8 corners of the bounding box
    corners = np.array([[-l/2, -h/2, -w/2],
                        [ l/2, -h/2, -w/2],
                        [ l/2,  h/2, -w/2],
                        [-l/2,  h/2, -w/2],
                        [-l/2, -h/2,  w/2],
                        [ l/2, -h/2,  w/2],
                        [ l/2,  h/2,  w/2],
                        [-l/2,  h/2,  w/2]])

    # Apply rotation (inverse of object rotation)
    rot_mat = np.array([
        [np.cos(angle), 0, -np.sin(angle)],
        [0, 1, 0],
        [np.sin(angle), 0, np.cos(angle)]
    ])
    rotated_corners = np.dot(corners, rot_mat.T)

    # Translate the corners to the box center
    rotated_corners += center

    # Define lines between corners to create the box edges
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
    ]
    lines = np.array(lines)

    # Create line set for visualization
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(rotated_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 1, 0])  # Green for the box edges

    return line_set


def load_lidar_bin(bin_path):
    """Load a KITTI .bin file."""
    pc = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pc[:, :3]  # Ignore intensity

def preprocess_frame(lidar_bin_path, label_path, calib_path):
    lidar_points = load_lidar_bin(lidar_bin_path)
    labels = read_label_file(label_path)
    calib = Calibration(calib_path)
    frustum_data = extract_frustum_data(lidar_points, labels, calib)
    return frustum_data
