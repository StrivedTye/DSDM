from torch.utils.data import Dataset
from data_classes import PointCloud, Box, BBox
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import os
import copy
import torch
from tqdm import tqdm
import kitty_utils as utils
from kitty_utils import getModel
from searchspace import KalmanFiltering
import logging
import json
from pandaset import DataSet as pandaset


class waymoDataset():

    def __init__(self, benchmark_list_dir, dataset_dir, input_size, offset_BB=0, scale_BB=1.0, category="vehicle"):
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB
        self.input_size = input_size

        self.benchmark_list_dir = benchmark_list_dir
        self.dataset_dir = dataset_dir
        self.category = category
        self.list_of_anno = []
        self.list_tracklet = self.getAllTracklet()

    def __len__(self):
        return len(self.list_tracklet)

    def isTiny(self):
        return False

    def __getitem__(self, item):
        PCs, BBs, ANNOs = self.getBBandPC(item)
        return PCs, BBs, ANNOs

    def getBBandPC(self, tracklet_id):
        tracklet_info = self.list_tracklet[tracklet_id]
        segment_name = tracklet_info['segment_name']
        start_frame = tracklet_info['frame_range'][0]
        end_frame = tracklet_info['frame_range'][1]
        object_id = tracklet_info['id']

        ego_info = np.load(os.path.join(self.dataset_dir, 'ego_info', '{:}.npz'.format(segment_name)),
                           allow_pickle=True)
        gt_info = np.load(os.path.join(self.dataset_dir, 'gt_info', '{:}.npz'.format(segment_name)),
                          allow_pickle=True)
        pc_all = np.load(os.path.join(self.dataset_dir, 'pc', 'clean_pc', '{:}.npz'.format(segment_name)),
                          allow_pickle=True)
        BBs = []
        PCs = []
        Annos = []
        for frame in range(start_frame, end_frame + 1):
            ego_matrix = ego_info[str(frame)]

            # get all boxes in the tracklet
            bboxes, ids = gt_info['bboxes'][frame], gt_info['ids'][frame]
            index = ids.index(object_id)
            cur_box = bboxes[index] # the order in every element is x,y,z,theta,l,w,h
            cur_box = BBox.array2bbox(cur_box)
            cur_box = BBox.bbox2world(ego_matrix, cur_box)

            center = [cur_box.x, cur_box.y, cur_box.z]
            size = [cur_box.w, cur_box.l, cur_box.h]
            orientation = Quaternion(axis=[0, 0, 1], radians=cur_box.o) # x-axis -> length; y->width, z->height
            BB = Box(center, size, orientation)
            BBs.append(BB)

            # get point clouds
            pc = pc_all[str(frame)]
            pc = PointCloud(pc.T)
            pc.transform(ego_matrix)
            PCs.append(pc)

            # get anno
            anno = {'occluded': 0, 'scene': segment_name, 'track_id':ids}
            Annos.append(anno)
        return PCs, BBs, Annos

    def getAllTracklet(self):
        bench_list_json = os.path.join(self.benchmark_list_dir, self.category, 'bench_list.json')
        with open(bench_list_json, 'r') as f:
            list_of_tracklet = json.load(f)

        exist_files = os.listdir(os.path.join(self.dataset_dir, 'ego_info'))
        exist_npz_files = [x[0:-4] for x in exist_files if 'npz' in x]

        final_list_of_tracklet = []
        for tracklet_info in list_of_tracklet:

            start_frame = tracklet_info['frame_range'][0]
            end_frame = tracklet_info['frame_range'][1]

            if tracklet_info['segment_name'] in exist_npz_files:
                final_list_of_tracklet.append(tracklet_info)
                self.list_of_anno.extend([frame for frame in range(start_frame, end_frame + 1)])

        return final_list_of_tracklet


class pandaDataset():

    def __init__(self, path):
        self.dataset = pandaset(path)

    def getSceneID(self, split):
        if "TRAIN" in split.upper():  # Training SET
            if "TINY" in split.upper():
                sceneID = [1]
            else:
                sceneID = list(range(1, 100))
        elif "VALID" in split.upper():  # Validation Set
            if "TINY" in split.upper():
                sceneID = [80]
            else:
                sceneID = list(range(100, 110))
        elif "TEST" in split.upper():  # Testing Set
            if "TINY" in split.upper():
                sceneID = [110]
            else:
                sceneID = list(range(110, 124))

        else:  # Full Dataset
            sceneID = list(range(124))
        return sceneID

    def getBBandPC(self, anno):

        PC, box = self.getPCandBBfromPandas(anno)
        return PC, box

    def getListOfAnno(self, sceneID, category_name="Car", train=True):
        list_of_scene = [scene for scene in self.dataset.sequences() if int(scene) in sceneID]
        list_of_tracklet_anno = []

        if train:
            ratio = 5 # ratio = 4 # select one frame with the interval of 5
            for scene in list_of_scene:

                cur_scene = self.dataset[scene]
                cur_scene.load_cuboids()

                box_all = cur_scene.cuboids[::ratio]
                for frame, box in enumerate(box_all):
                    box.insert(loc=0, column="frame", value=frame*ratio)

                df = pd.concat(box_all, axis=0)
                df = df[df["label"] == category_name]
                # df = df[df['attributes.object_motion'] == 'Moving']
                df.insert(loc=0, column="scene", value=scene)

                list_track_id = sorted(df.uuid.unique())[::8] #[::2]
                for track_id in list_track_id:
                    df_tracklet = df[df["uuid"] == track_id]
                    df_tracklet = df_tracklet.reset_index(drop=True)
                    tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                    list_of_tracklet_anno.append(tracklet_anno)
        else:
            for scene in list_of_scene:

                cur_scene = self.dataset[scene]
                cur_scene.load_cuboids()

                box_all = cur_scene.cuboids
                for frame, box in enumerate(box_all):
                    box.insert(loc=0, column="frame", value=frame)

                df = pd.concat(box_all, axis=0) # all frames

                df = df[df["label"] == category_name]
                df = df[df['attributes.object_motion'] == 'Moving']

                df.insert(loc=0, column="scene", value=scene)
                df = df.rename(columns={'label': 'type'})

                list_track_id = sorted(df.uuid.unique())[::3]
                for track_id in list_track_id:
                    df_tracklet = df[df["uuid"] == track_id]
                    df_tracklet = df_tracklet.reset_index(drop=True)
                    # if df_tracklet['attributes.object_motion'][0] == 'Moving': continue
                    tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                    list_of_tracklet_anno.append(tracklet_anno)

        return list_of_tracklet_anno

    def getPCandBBfromPandas(self, box):
        center = [box["position.x"], box["position.y"], box["position.z"]]
        size = [box["dimensions.x"], box["dimensions.y"], box["dimensions.z"]]
        orientation = Quaternion(
            axis=[0, 0, 1], radians=box["yaw"]) * Quaternion(axis=[0, 0, 1], radians=np.pi / 2)
        BB = Box(center, size, orientation)

        try:
            # VELODYNE PointCloud
            fp = os.path.join(self.dataset._directory, box["scene"], 'lidar', '{:02}.pkl.gz'.format(box["frame"]))
            PC = pd.read_pickle(fp)
            PC = PC.values
            PC = PointCloud(PC[:, 0:3].T)
        except :
            # in case the Point cloud is missing
            PC = PointCloud(np.array([[0, 0, 0]]).T)

        return PC, BB


class kittiDataset():

    def __init__(self, path):
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")

    def getSceneID(self, split):
        if "TRAIN" in split.upper():  # Training SET
            if "TINY" in split.upper():
                sceneID = [0]
            else:
                sceneID = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set
            if "TINY" in split.upper():
                sceneID = [18]
            else:
                sceneID = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            if "TINY" in split.upper():
                sceneID = [19]
            else:
                sceneID = list(range(19, 21))

        else:  # Full Dataset
            sceneID = list(range(21))
        return sceneID

    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib',
                                  anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        PC, box = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box

    def getListOfAnno(self, sceneID, category_name="Car"):
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
            int(path) in sceneID
        ]
        # print(self.list_of_scene)
        list_of_tracklet_anno = []
        for scene in list_of_scene:

            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            if category_name == "All":
                df0 = df[df["type"] == "Car"]
                df1 = df[df["type"] == "Pedestrian"]
                df2 = df[df["type"] == "Van"]
                df3 = df[df["type"] == "Cyclist"]
                df = pd.concat([df0, df2, df1, df3], axis=0)
            else:
                df = df[df["type"] == category_name]

            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                list_of_tracklet_anno.append(tracklet_anno)

        return list_of_tracklet_anno

    def getPCandBBfromPandas(self, box, calib):
        center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
        size = [box["width"], box["length"], box["height"]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(center, size, orientation)

        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
                                         '{:06}.bin'.format(box["frame"]))
            PC = PointCloud(
                np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            PC.transform(calib)
        except :
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin)
            PC = PointCloud(np.array([[0, 0, 0]]).T)

        return PC, BB

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data


class SiameseDataset(Dataset):

    def __init__(self,
                 input_size,
                 path,
                 split,
                 category_name="Car",
                 regress="GAUSSIAN",
                 offset_BB=0,
                 scale_BB=1.0,
                 dataset="KITTI", train=True):

        self.train = train

        if dataset=="KITTI":
            self.dataset = kittiDataset(path=path)
        elif dataset=="PANDA":
            self.dataset = pandaDataset(path=path)

        self.input_size = input_size
        self.split = split
        self.sceneID = self.dataset.getSceneID(split=split)
        self.getBBandPC = self.dataset.getBBandPC

        self.category_name = category_name
        self.regress = regress

        if dataset=="KITTI":
            self.list_of_tracklet_anno = self.dataset.getListOfAnno(
                self.sceneID, category_name)
        elif dataset=="PANDA":
            self.list_of_tracklet_anno = self.dataset.getListOfAnno(
                self.sceneID, category_name, self.train)

        self.list_of_anno = [
            anno for tracklet_anno in self.list_of_tracklet_anno
            for anno in tracklet_anno
        ]

    def isTiny(self):
        return ("TINY" in self.split.upper())

    def __getitem__(self, index):
        return self.getitem(index)


class SiameseTrain(SiameseDataset):

    def __init__(self,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 regress="GAUSSIAN",
                 sigma_Gaussian=1,
                 offset_BB=0,
                 scale_BB=1.0, dataset="KITTI", train=True):
        super(SiameseTrain,self).__init__(
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            regress=regress,
            offset_BB=offset_BB,
            scale_BB=scale_BB, dataset=dataset, train=train)

        self.dataset = dataset
        self.sigma_Gaussian = sigma_Gaussian
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

        self.num_candidates_perframe = 4 #4

        logging.info("preloading PC...")
        self.list_of_PCs = [None] * len(self.list_of_anno)
        self.list_of_BBs = [None] * len(self.list_of_anno)
        for index in tqdm(range(len(self.list_of_anno))):
            anno = self.list_of_anno[index]
            PC, box = self.getBBandPC(anno)
            new_PC = utils.cropPC(PC, box, offset=10)

            self.list_of_PCs[index] = new_PC
            self.list_of_BBs[index] = box
        logging.info("PC preloaded!")

        logging.info("preloading Model..")
        self.model_PC = [None] * len(self.list_of_tracklet_anno)
        for i in tqdm(range(len(self.list_of_tracklet_anno))):
            list_of_anno = self.list_of_tracklet_anno[i]
            PCs, BBs = [], []
            cnt = 0
            for anno in list_of_anno:
                # this_PC, this_BB = self.getBBandPC(anno)
                # PCs.append(this_PC)
                # BBs.append(this_BB)
                anno["model_idx"] = i
                anno["relative_idx"] = cnt
                cnt += 1

            # self.model_PC[i] = getModel(
            #     PCs, BBs, offset=self.offset_BB, scale=self.scale_BB)

        logging.info("Model preloaded!")

    def __getitem__(self, index):
        return self.getitem(index)

    def getPCandBBfromIndex(self, anno_idx):
        this_PC = self.list_of_PCs[anno_idx]
        this_BB = self.list_of_BBs[anno_idx]
        return this_PC, this_BB

    def getitem(self, index):
        anno_idx = self.getAnnotationIndex(index)
        sample_idx = self.getSearchSpaceIndex(index)

        if sample_idx == 0:
            sample_offsets = np.zeros(3)
        else:
            gaussian = KalmanFiltering(bnd=[0.5, 0.5, 0.2, 5])
            sample_offsets = gaussian.sample(1)[0]
            # if self.dataset == 'PANDA':
            #     sample_offsets = np.random.uniform(low=-1, high=1, size=3)
            #     sample_offsets[0] = sample_offsets[0]*1.5
            #     sample_offsets[1] = sample_offsets[1]*0.1
            #     sample_offsets[2] = np.random.uniform(low=-1, high=1, size=1)[0]*5

        this_anno = self.list_of_anno[anno_idx]

        this_PC, this_BB = self.getPCandBBfromIndex(anno_idx)
        sample_BB = utils.getOffsetBB(this_BB, sample_offsets)

        # sample_PC = utils.cropAndCenterPC(
        #     this_PC, sample_BB, offset=self.offset_BB, scale=self.scale_BB)
        sample_PC, sample_origin_state, sample_gt_state, sample_semantic= utils.cropAndCenterPC_label(
            this_PC, sample_BB, this_BB, sample_offsets, offset=self.offset_BB, scale=self.scale_BB)
        if sample_PC.nbr_points() <= 20:
            return self.getitem(np.random.randint(0, self.__len__()))
        # sample_PC = utils.regularizePC(sample_PC, self.input_size)[0]

        num_seed = 3
        seeds_offset = np.random.uniform(low=-0.3, high=0.3, size=[num_seed,4])
        seeds_offset[:, -2] = seeds_offset[:, -2] * 0.5
        # seeds_offset[:, -1] = seeds_offset[:, -1] * 5.0 # radian,don't need multiply 5
        parallel_seeds = np.reshape(sample_gt_state, (1, 7))
        parallel_seeds = np.repeat(parallel_seeds, num_seed+1, 0) #[63+1, 7]
        parallel_seeds[1:, [0,1,2,6]] = parallel_seeds[1:, [0,1,2,6]] + seeds_offset
        # parallel_seeds[0] = sample_gt_state

        # sample_origin_state = torch.from_numpy(sample_origin_state).float()
        parallel_seeds = torch.from_numpy(parallel_seeds).float()
        sample_gt_state = torch.from_numpy(sample_gt_state).float()
        sample_PC, sample_semantic = utils.regularizePCwithlabel(sample_PC, sample_semantic, self.input_size) #[N, 3], [N]

        if this_anno["relative_idx"] == 0:
            prev_idx = 0
            fir_idx = 0
        else:
            prev_idx = anno_idx - 1
            fir_idx = anno_idx - this_anno["relative_idx"]
        gt_PC_pre, gt_BB_pre = self.getPCandBBfromIndex(prev_idx)
        gt_PC_fir, gt_BB_fir = self.getPCandBBfromIndex(fir_idx)

        if sample_idx == 0:
            samplegt_offsets = np.zeros(3)
        else:
            samplegt_offsets = np.random.uniform(low=-0.1, high=0.1, size=3) # Car:[0.1, 0.1]
            samplegt_offsets[2] = samplegt_offsets[2]*5.0
        gt_BB_pre = utils.getOffsetBB(gt_BB_pre, samplegt_offsets)

        gt_PC = getModel([gt_PC_fir,gt_PC_pre], [gt_BB_fir,gt_BB_pre], offset=self.offset_BB, scale=self.scale_BB)

        # generate semantic label
        gt_BB_fir_norm = utils.Centerbox(gt_BB_fir, gt_BB_fir)
        gt_semantic = utils.getlabelPC(gt_PC, gt_BB_fir_norm)

        if gt_PC.nbr_points() <= 20:
            return self.getitem(np.random.randint(0, self.__len__()))
        # gt_PC = utils.regularizePC(gt_PC, self.input_size)
        gt_PC, gt_semantic = utils.regularizePCwithlabel(gt_PC, gt_semantic, int(self.input_size))
        gt_origin_state = np.array([0, 0, 0, gt_BB_fir.wlh[1], gt_BB_fir.wlh[0], gt_BB_fir.wlh[2], 0])  # [x, y, z, l, w, h, theta]
        gt_origin_state = torch.from_numpy(gt_origin_state).float()

        return sample_PC, parallel_seeds, sample_gt_state, sample_semantic, gt_PC, gt_origin_state, gt_semantic

    def __len__(self):
        nb_anno = len(self.list_of_anno)
        return nb_anno * self.num_candidates_perframe

    def getAnnotationIndex(self, index):
        return int(index / (self.num_candidates_perframe))

    def getSearchSpaceIndex(self, index):
        return int(index % self.num_candidates_perframe)


class SiameseTest(SiameseDataset):

    def __init__(self,
                 input_size,
                 path,
                 split="",
                 category_name="Car",
                 regress="GAUSSIAN",
                 offset_BB=0,
                 scale_BB=1.0, dataset="KITTI", train=False):
        super(SiameseTest,self).__init__(
            input_size=input_size,
            path=path,
            split=split,
            category_name=category_name,
            regress=regress,
            offset_BB=offset_BB,
            scale_BB=scale_BB, dataset=dataset, train=train)
        self.split = split
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

    def getitem(self, index):
        list_of_anno = self.list_of_tracklet_anno[index]
        PCs = []
        BBs = []
        for anno in list_of_anno:
            this_PC, this_BB = self.getBBandPC(anno)
            PCs.append(this_PC)
            BBs.append(this_BB)
        return PCs, BBs, list_of_anno

    def __len__(self):
        return len(self.list_of_tracklet_anno)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from data_classes import Box, PointCloud

    # dataset_path = '/home/tye/study/visualTraking/Database/PandaSet'
    # test = SiameseTest(1024, dataset_path, split="Test", dataset="PANDA", train=False, category_name='Car')

    # test = nuscenes(path='/home/tye/study/visualTraking/Database/v1.0-mini',
    #                 input_size=1024,
    #                 offset_BB=0,
    #                 scale_BB=1.25)

    dataset_path = '/workspace/data/tracking/test_datasets/waymo_valid_extract/'
    benchmark_dir = dataset_path + 'benchmark'
    test = waymoDataset(benchmark_dir, dataset_path)

    print(len(test))
    PCs, BBs, _ = test[0]

    for i in range(len(BBs)-1):
        new_gt_BB = utils.Centerbox(BBs[i+1], BBs[i])
        angle = new_gt_BB.orientation.yaw_pitch_roll[0]
        degree = angle / np.pi * 180
        offset = [new_gt_BB.center[0], new_gt_BB.center[1], degree]

        print(offset)

    # exit()
    # Create figure for TRACKING
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    # Create axis in 3D
    ax = fig.gca(projection='3d')

    ratio=2
    color_list = ["red", "blue", "green", "black", "yellow", "cyan", "magenta"]
    num = len(PCs) if len(PCs) < len(color_list) else len(color_list)
    for i in range(num):
        this_PC, this_BB = PCs[i+len(PCs)-10], BBs[i+len(PCs)-10]
        this_PC = utils.cropPC(this_PC, this_BB, offset=10)

        ax.scatter(
            this_PC.points[0, ::ratio],
            this_PC.points[1, ::ratio],
            this_PC.points[2, ::ratio],
            s=5, color=color_list[int(i)])

        order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]
        # # Plot Box
        ax.plot(
            this_BB.corners()[0, order],
            this_BB.corners()[1, order],
            this_BB.corners()[2, order],
            color=color_list[int(i)],
            alpha=0.5,
            linewidth=2,
            linestyle="-")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

if __name__ == '__main__0':

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from data_classes import Box, PointCloud

    dataset_path = '/home/tye/study/visualTraking/Database/PandaSet'
    train = SiameseTrain(1024, dataset_path, split="Train_tiny", dataset="PANDA")
    sample_PC, sample_origin_state, sample_gt_state, sample_semantic, gt_PC, gt_origin_state, gt_semantic = train[185]
    view_PC = PointCloud(sample_PC.numpy().T)
    print(sample_origin_state[2])
    print(sample_gt_state)
    sample_BB_ = utils.lidar2bbox([sample_origin_state[2].numpy()])[0]
    gt_BB_ = utils.lidar2bbox([sample_gt_state.numpy()])[0]


    from metrics import estimateOverlap, estimateAccuracy, Success, Precision
    a = estimateOverlap(sample_BB_, gt_BB_)

    sample_BB_.rotate(Quaternion(axis=[1, 0, 0], angle=-np.pi/2))
    gt_BB_.rotate(Quaternion(axis=[1, 0, 0], angle=-np.pi / 2))

    # new_this_BB = copy.deepcopy(gt_BB_)
    # new_ref_BB = copy.deepcopy(sample_BB_)
    # degree = new_this_BB.orientation.degrees - new_ref_BB.orientation.degrees
    # print(degree)
    new_gt_BB = utils.Centerbox(gt_BB_, sample_BB_)
    angle = new_gt_BB.orientation.yaw_pitch_roll[0]
    degree = angle / np.pi * 180

    offset = [new_gt_BB.center[0], new_gt_BB.center[1], degree]
    print(offset)

    # offset = sample_gt_state.numpy()[[0, 1, 6]]
    # offset[-1] = offset[-1]/np.pi * 180
    gt_BB_recon = utils.getOffsetBB(sample_BB_, offset)

    # Create figure for TRACKING
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    plt.rcParams['savefig.dpi'] = 300
    # Create axis in 3D
    ax = fig.gca(projection='3d')

    # Scatter plot the cropped point cloud
    ratio = 1
    ax.scatter(
        view_PC.points[0, ::ratio],
        view_PC.points[1, ::ratio],
        view_PC.points[2, ::ratio],
        s=3,
        c=view_PC.points[2, ::ratio])

    flag = np.reshape(sample_semantic, (1, -1)).repeat(3, 1)
    pc = np.where(flag == 1., view_PC.points, 0)
    ax.scatter(
        pc[0, ::ratio],
        pc[1, ::ratio],
        pc[2, ::ratio],
        s=20, color='blue')


    # point order to draw a full Box
    order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]

    # # Plot Box
    ax.plot(
        sample_BB_.corners()[0, order],
        sample_BB_.corners()[1, order],
        sample_BB_.corners()[2, order],
        color="green",
        alpha=0.5,
        linewidth=2,
        linestyle="-")

    ax.plot(
        gt_BB_.corners()[0, order],
        gt_BB_.corners()[1, order],
        gt_BB_.corners()[2, order],
        color="red",
        alpha=0.5,
        linewidth=2,
        linestyle="-")


    ax.plot(
        gt_BB_recon.corners()[0, order],
        gt_BB_recon.corners()[1, order],
        gt_BB_recon.corners()[2, order],
        color="blue",
        alpha=0.5,
        linewidth=2,
        linestyle="-")


    # ax.set_axis_off()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
