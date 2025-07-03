import os

def list_frame_ids(lidar_dir):
    # List all .bin files in the lidar_dir and extract frame IDs
    frame_ids = [f.split('.')[0] for f in os.listdir(lidar_dir) if f.endswith('.bin')]
    frame_ids.sort()  # Sort numerically, if needed
    return frame_ids

# Path to your LIDAR point clouds directory (test data)
lidar_dir = '/home/dfki.uni-bremen.de/sshete/FrustrumPointNet/data/lidar'  # Update this path as needed


# Get list of frame ids from the lidar directory
frame_ids = list_frame_ids(lidar_dir)

# Define the base directory from where LIDAR data is loaded (use lidar_dir)
base_dir = os.path.dirname(lidar_dir)

# Define file paths for the train, validation, and test splits
train_split_path = os.path.join(base_dir, 'train_split.txt')  # Save in the same directory as lidar_dir
val_split_path = os.path.join(base_dir, 'val_split.txt')
test_split_path = os.path.join(base_dir, 'test_split.txt')  # Save in the same directory as test_lidar_dir

# Create the directories if they don't exist
os.makedirs(os.path.dirname(train_split_path), exist_ok=True)
os.makedirs(os.path.dirname(val_split_path), exist_ok=True)
os.makedirs(os.path.dirname(test_split_path), exist_ok=True)

# Split the data into train, val, and test sets (e.g., 80-10-10 split)
train_size = int(0.8 * len(frame_ids))
val_size = int(0.1 * len(frame_ids))
test_size = len(frame_ids) - train_size - val_size

train_frames = frame_ids[:train_size]
val_frames = frame_ids[train_size:train_size + val_size]
test_frames = frame_ids[train_size + val_size:]

# Write to train split file
with open(train_split_path, 'w') as f:
    for frame in train_frames:
        f.write(f"{frame}\n")

# Write to validation split file
with open(val_split_path, 'w') as f:
    for frame in val_frames:
        f.write(f"{frame}\n")

# Write to test split file
with open(test_split_path, 'w') as f:
    for frame in test_frames:
        f.write(f"{frame}\n")

# Print confirmation
print(f"Train split written to {train_split_path}")
print(f"Val split written to {val_split_path}")
print(f"Test split written to {test_split_path}")
