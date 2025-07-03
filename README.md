# Frustum PointNet for 3D Object Detection

This repository implements **Frustum PointNet**, a 3D deep learning model for object detection and segmentation on LiDAR point clouds, tailored for tasks like autonomous driving. It performs semantic segmentation, 3D bounding box estimation, and visualization of point cloud data.

---

## Features

- Train and evaluate Frustum PointNet on LiDAR datasets  
- Semantic segmentation of point clouds  
- 3D bounding box regression (center, size, angle)  
- Visualization with Open3D for predicted and ground truth bounding boxes  
- Support for KITTI-style dataset format (LiDAR, calibration, and labels)  
- Evaluation metrics including segmentation accuracy and IoU  

---
### Project Structure
```
├── checkpoints/
├── models/
│   └── frustrum_pointnet.py    # Model definition
│
├── datasets/
│   └── frustrum_dataset.py     # Dataset loading
├── data/
│   ├── lidar/
│   ├──lidar/
│   ├──calib/
│
├── utils/
│   ├── visualize.py            # Visualization utilities
│   ├── metrics.py              # Metric computations
│
├── trainer.py                  # Training loop
├── eval.py                     # Evaluation script
├── test.py                     # Test with metrics
├── visualize.py                # Visualization of predictions
├── requirements.txt            # Required Python packages
└── README.md                   # This file
```
### Dependencies
Python 3.7+
Pytorch
Open3D
NumPy
tqdm

### Dataset
This project uses the KITTI 3D object detection dataset or a similar frustum-based point cloud dataset.

Where to download
You can download the KITTI dataset from the official site:
http://www.cvlibs.net/datasets/kitti/

Specifically, you need the following parts:
Velodyne point clouds (lidar data)
Labels (3D bounding boxes annotations)
Calibration files (camera calibration and LiDAR-to-camera transformation)

Dataset folder structure
Organize your dataset files in the following directory layout:

```
data/
├── calib/      # Calibration files (.txt)
├── label/      # Label files (.txt) with ground truth 3D boxes
└── lidar/      # Velodyne point cloud files (.bin)
```
Make sure the files in each folder are named consistently so they correspond across folders (e.g., 000001.bin in lidar corresponds to 000001.txt in label and calib).

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/frustum-pointnet.git
   cd frustum-pointnet

2. Create and activate a Python virtual environment (recommended):
```
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```
3. Install dependencies:

pip install -r requirements.txt

4. Make sure you have CUDA installed if you want GPU acceleration.

## Usage
### Train the Model
1. Use trainer.py to train the model:
python trainer.py --epochs 300 --batch_size 64 --lr 0.001

2. Evaluate the Model
Run evaluation on validation or test split with:

python eval.py --model_path checkpoints/Frustum_best_model.pth --split val

3. Test Script
For detailed metrics on a test set:

python test.py

4. Visualize Predictions
Visualize predicted segmentation and bounding boxes with Open3D:

python visualize.py


#### Citation
```
@inproceedings{qi2018frustum,
  title={Frustum PointNets for 3D Object Detection from RGB-D Data},
  author={Qi, Charles R and Liu, Wei and Wu, Chenxia and Su, Hao and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```
