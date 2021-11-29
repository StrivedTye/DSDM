#python train_tracking_eval.py --data_dir=/workspace/data/tracking/train_datasets/KITTI_Tracking/truly_using/ --save_root_dir=results1 --roi_voxels=8 --batchSize=32
#python train_tracking_eval.py --data_dir=/workspace/data/tracking/train_datasets/KITTI_Tracking/truly_using/ --save_root_dir=results2 --roi_voxels=5
#python train_tracking_eval.py --data_dir=/workspace/data/tracking/train_datasets/KITTI_Tracking/truly_using/ --save_root_dir=results3 --roi_voxels=6
python train_tracking_eval.py --data_dir=/workspace/data/tracking/train_datasets/KITTI_Tracking/truly_using/ --save_root_dir=results_ped --category_name=Pedestrian
python train_tracking_eval.py --data_dir=/workspace/data/tracking/train_datasets/KITTI_Tracking/truly_using/ --save_root_dir=results_van --category_name=Van
python train_tracking_eval.py --data_dir=/workspace/data/tracking/train_datasets/KITTI_Tracking/truly_using/ --save_root_dir=results_cyc --category_name=Cyclist