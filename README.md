# DSDM
3D tracking in point cloud

### Install 
 + Env: torch1.1; cuda10.1.214
 + Dependencies: ``pip install -r requirement.txt``
 + Pointnet++: ``python setup.py install``
 + RoIAware_Pool3D: ``cd ./roiaware_pool3d ``, then ``python setup.py develop``

### Dataset
 + [KITTI](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip;http://www.cvlibs.net/download.php?file=data_tracking_calib.zip;http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip)
 + [Pandaset](https://pandaset.org/)
 + [Waymo](https://github.com/TuSimple/LiDAR_SOT) The LIDAR_SOT is an evaluation benchmark based on the waymo validation set.

### Train 
 + Run ``python train_tracking_eval.py --data_dir="your dataset path" --dataSize=24``

### Test
 + Run ``python test_tracking.py --data_dir="your dataset path" -save_root_dir="your checkpoint path" --model="which checkpoint"``
 + You can access the trained [model](https://drive.google.com/file/d/1KpCit2XgEaKWpT41ZEDsRiPLyjpCXEsw/view?usp=sharing)

### Acknowledgment
This project is based on [P2B](https://github.com/HaozheQi/P2B), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), and [LIDAR_SOT](https://github.com/TuSimple/LiDAR_SOT).
