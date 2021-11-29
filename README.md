# DSDM
3D tracking in point cloud

### Install 
 + Env: torch1.1; cuda10.1.214
 + Dependencies: ``pip install -r requirement.txt``
 + Pointnet++: ``python setup.py install``
 + RoIAware_Pool3D: ``cd ./roiaware_pool3d ``, then ``python setup.py develop``

### Train 
 + Run ``python train_tracking_eval.py --data_dir="your dataset path" --dataSize=24``

### Test
 + Run ``python test_tracking.py --data_dir="your dataset path" -save_root_dir="your checkpoint path" --model="which checkpoint"``

### Acknowledgment
This project is based on [P2B](https://github.com/HaozheQi/P2B) and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
