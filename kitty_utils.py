import torch
import os
import copy
import numpy as np
from pyquaternion import Quaternion
from data_classes import PointCloud, Box
from metrics import estimateOverlap
from scipy.optimize import leastsq


def decode_box_BinBased(output, candidates,
                        loc_scope=1.5,
                        loc_bin_size=0.5,
                        num_head_bin=5):

    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2

    cla_x = output[:, 0:per_loc_bin_num]
    cla_z = output[:, per_loc_bin_num:per_loc_bin_num * 2]
    reg_x = output[:, per_loc_bin_num * 2:per_loc_bin_num * 3]
    reg_z = output[:, per_loc_bin_num * 3:per_loc_bin_num * 4]
    cla_angle = output[:, per_loc_bin_num * 4:per_loc_bin_num * 4 + num_head_bin]
    reg_angle = output[:, per_loc_bin_num * 4 + num_head_bin:]

    ##################################### decode x,z
    x_bin = np.argmax(cla_x, 1)
    z_bin = np.argmax(cla_z, 1)

    x_bin_max = np.max(cla_x, 1)
    z_bin_max = np.max(cla_z, 1)

    x_bin_flag = cla_x >= x_bin_max
    z_bin_flag = cla_z >= z_bin_max

    # get regression value correspondding to the maximum bin
    x_bin_reg = reg_x[x_bin_flag]
    z_bin_reg = reg_z[z_bin_flag]

    # decode: get x_gt-x_candi
    delta_x = x_bin_reg * loc_bin_size \
              + (x_bin * loc_bin_size + loc_bin_size / 2) - loc_scope

    delta_z = z_bin_reg * loc_bin_size \
              + (z_bin * loc_bin_size + loc_bin_size / 2) - loc_scope

    ####################################### decode angle
    #
    angle_bin = np.argmax(cla_angle, 1)
    angle_bin_max = np.max(cla_angle, 1)
    angle_bin_flag = cla_angle >= angle_bin_max
    angle_bin_reg = reg_angle[angle_bin_flag]

    angle_per_class = (np.pi / 6) / num_head_bin

    delta_angle = angle_bin_reg * (angle_per_class / 2) \
                  + (angle_bin * angle_per_class + angle_per_class / 2) - np.pi/12

    offsets = -np.stack([delta_x, delta_z, delta_angle]).T

    final_box=[]
    for box, offset in zip(candidates, offsets):
        final_box.append(getOffsetBB(box, offset))
    return final_box

def enlarge_box3d(boxes3d, extra_width=1):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 1] += extra_width
    return large_boxes3d

def lidar2bbox(boxs_para):
    # box: [[x, y, z, l, w, h, ry]] canonical coord, shape of [n,7]
    bboxs = []

    for b in boxs_para:
        center = [b[0], b[1], b[2]]
        size = [b[4], b[3], b[5]] #wlh
        orientation = Quaternion(axis=[0, 0, 1], angle=b[6]) #* Quaternion(axis=[1, 0, 0], angle=np.pi/2)
        bbox = Box(center, size, orientation)
        bbox.rotate(Quaternion(axis=[1, 0, 0], angle=np.pi/2))
        bboxs.append(bbox)

    return bboxs


def warp_Box(box, offset):
    assert len(offset) == 3

    new_box = copy.deepcopy(box)
    # if offset[0]>new_box.wlh[0]:
    #     offset[0] = np.random.uniform(-1,1)
    # if offset[1]>min(new_box.wlh[1],2):
    #     offset[1] = np.random.uniform(-1,1)

    ################ First: translate
    T = np.array([offset[0], 0, offset[1]]) # not yet canonial
    # T = np.array([offset[0], offset[1], 0]) #
    new_box.translate(T)

    ################Second: rotation
    # back to local coodinate
    rot_quat = Quaternion(matrix=new_box.rotation_matrix)
    trans = np.array(new_box.center)
    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)

    angle = offset[2] * np.pi / 180
    new_box.rotate(Quaternion(axis=[0, 0, 1],
                              angle=angle))


    ################Re-back
    new_box.rotate(rot_quat)
    new_box.translate(trans)

    return new_box


def distanceBB_Gaussian(box1, box2, sigma=1):
    off1 = np.array([
        box1.center[0], box1.center[2],
        Quaternion(matrix=box1.rotation_matrix).degrees
    ])
    off2 = np.array([
        box2.center[0], box2.center[2],
        Quaternion(matrix=box2.rotation_matrix).degrees
    ])
    dist = np.linalg.norm(off1 - off2)
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return score


# IoU or Gaussian score map
def getScoreGaussian(offset, sigma=1):
    coeffs = [1, 1, 1 / 5]
    dist = np.linalg.norm(np.multiply(offset, coeffs))
    score = np.exp(-0.5 * (dist) / (sigma * sigma))
    return torch.tensor([score])


def getScoreIoU(a, b):
    score = estimateOverlap(a, b)
    return torch.tensor([score])


def getScoreHingeIoU(a, b):
    score = estimateOverlap(a, b)
    if score < 0.5:
        score = 0.0
    return torch.tensor([score])


def getOffsetBB(box, offset):
    rot_quat = Quaternion(matrix=box.rotation_matrix)
    trans = np.array(box.center)

    new_box = copy.deepcopy(box)

    new_box.translate(-trans)
    new_box.rotate(rot_quat.inverse)

    if offset[0]>new_box.wlh[0]:
        offset[0] = np.random.uniform(-1,1)
    if offset[1]>min(new_box.wlh[1],2):
        offset[1] = np.random.uniform(-1,1)

    # REMOVE TRANSfORM
    if len(offset) == 3:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180))

        new_box.translate(np.array([offset[0], offset[1], 0]))

    elif len(offset) == 4:
        new_box.rotate(
            Quaternion(axis=[0, 0, 1], angle=offset[3] * np.pi / 180))

        new_box.translate(np.array([offset[0], offset[1], offset[2]]))

    # APPLY PREVIOUS TRANSFORMATION
    new_box.rotate(rot_quat)
    new_box.translate(trans)
    return new_box


def voxelize(PC, dim_size=[48, 108, 48]):
    # PC = normalizePC(PC)
    if np.isscalar(dim_size):
        dim_size = [dim_size] * 3
    dim_size = np.atleast_2d(dim_size).T
    PC = (PC + 0.5) * dim_size
    # truncate to integers
    xyz = PC.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dim_size), 0)
    xyz = xyz[:, valid_ix]
    out = np.zeros(dim_size.flatten(), dtype=np.float32)
    out[tuple(xyz)] = 1
    # print(out)
    return out


def regularizePC2(PC, input_size, istrain=True):
    return regularizePC(PC=PC, input_size=input_size*2, istrain=istrain)


def regularizePC(PC, input_size, istrain=True):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != input_size:
            if not istrain:
                np.random.seed(1)
            new_pts_idx = np.random.randint(
                low=0, high=PC.shape[1], size=input_size, dtype=np.int64)
            PC = PC[:, new_pts_idx]
        PC = PC.reshape((3, input_size)).T

    else:
        PC = np.zeros((3, input_size)).T

    return torch.from_numpy(PC).float()

def regularizePCwithlabel(PC,label, input_size, istrain=True):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != input_size:
            if not istrain:
                np.random.seed(1)
            new_pts_idx = np.random.randint(
                low=0, high=PC.shape[1], size=input_size, dtype=np.int64)
            PC = PC[:, new_pts_idx]
            label = label[new_pts_idx]
        PC = PC.reshape((3, input_size)).T

    else:
        PC = np.zeros((3, input_size)).T

    return torch.from_numpy(PC).float(), torch.from_numpy(label).float()

def getModel(PCs, boxes, offset=0, scale=1.0, normalize=False):

    if len(PCs) == 0:
        return PointCloud(np.ones((3, 0)))
    points = np.ones((PCs[0].points.shape[0], 0))

    count = 0
    for PC, box in zip(PCs, boxes):
        if count == 0:
            # cropped_PC = cropAndCenterPC(PC, box, offset=offset+2.0, scale=scale, normalize=normalize)
            cropped_PC = cropAndCenterPC(PC, box, offset=offset, scale=scale, normalize=normalize)
        else:
            cropped_PC = cropAndCenterPC(PC, box, offset=offset, scale=scale, normalize=normalize)
        # try:
        if cropped_PC.points.shape[1] > 0:
            points = np.concatenate([points, cropped_PC.points], axis=1)

        count += 1
    PC = PointCloud(points)

    return PC

def cropPC(PC, box, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    return new_PC

def getlabelPC(PC, box, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    new_PC = PointCloud(PC.points.copy())
    rot_mat = np.transpose(box_tmp.rotation_matrix)
    trans = -box_tmp.center

    # align data
    new_PC.translate(trans)
    box_tmp.translate(trans)
    new_PC.rotate((rot_mat))
    box_tmp.rotate(Quaternion(matrix=(rot_mat)))
    
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = new_PC.points[0, :] < maxi[0]
    x_filt_min = new_PC.points[0, :] > mini[0]
    y_filt_max = new_PC.points[1, :] < maxi[1]
    y_filt_min = new_PC.points[1, :] > mini[1]
    z_filt_max = new_PC.points[2, :] < maxi[2]
    z_filt_min = new_PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_label = np.zeros(new_PC.points.shape[1])
    new_label[close] = 1
    return new_label

def cropPCwithlabel(PC, box,label, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    new_label = label[close]
    return new_PC,new_label

def weight_process(include,low,high):
    if include<low:
        weight = 0.7
    elif include >high:
        weight = 1
    else:
        weight = (include*2.0+3.0*high-5.0*low)/(5*(high-low))
    return weight

def func(a, x):
    k, b = a
    return k * x + b

def dist(a, x, y):
    return func(a, x) - y

def weight_process2(k):
    k = abs(k)
    if k>1:
        weight = 0.7
    else:
        weight = 1-0.3*k
    return weight

def cropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset, scale=scale)

    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC

def Centerbox(sample_box, gt_box):
    rot_mat = np.transpose(gt_box.rotation_matrix)
    trans = -gt_box.center

    new_box = copy.deepcopy(sample_box)
    new_box.translate(trans)
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    return new_box

def cropAndCenterPC_label(PC, sample_box, gt_box, sample_offsets, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(sample_box)

    new_label = getlabelPC(new_PC, gt_box, offset=offset, scale=scale)
    new_box_gt = copy.deepcopy(gt_box)
    # new_box_gt2 = copy.deepcopy(gt_box)

    #rot_quat = Quaternion(matrix=new_box.rotation_matrix)
    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)     
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    new_box_gt.translate(trans)
    new_box_gt.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+2.0, scale=1 * scale)
    #new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+0.6, scale=1 * scale)

    origin_state = np.array([0, 0, 0, new_box.wlh[1], new_box.wlh[0], new_box.wlh[2], 0]) #[x, y, z, l, w, h, theta]
    gt_state = np.array([new_box_gt.center[0], new_box_gt.center[1], new_box_gt.center[2],
                         new_box_gt.wlh[1], new_box_gt.wlh[0], new_box_gt.wlh[2],
                         -sample_offsets[2] * np.pi / 180]) # radian, not degree

    if normalize:
        new_PC.normalize(sample_box.wlh)

    return new_PC, origin_state, gt_state, new_label

def cropAndCenterPC_label_test_time(PC, sample_box, offset=0, scale=1.0):

    new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(sample_box)

    rot_quat = Quaternion(matrix=new_box.rotation_matrix)
    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)     
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset+2.0, scale=scale)

    return new_PC

def cropAndCenterPC_label_test(PC, sample_box, gt_box, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, sample_box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(sample_box)

    new_label = getlabelPC(new_PC, gt_box, offset=offset, scale=scale)
    new_box_gt = copy.deepcopy(gt_box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)     
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    new_box_gt.translate(trans)
    new_box_gt.rotate(Quaternion(matrix=(rot_mat)))
    # new_box_gt2.translate(trans)
    # new_box_gt2.rotate(rot_quat.inverse)

    # crop around box
    new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+2.0, scale=1 * scale)
    #new_PC, new_label = cropPCwithlabel(new_PC, new_box, new_label, offset=offset+0.6, scale=1 * scale)

    if normalize:
        new_PC.normalize(sample_box.wlh)
    return new_PC, new_label, new_box, new_box_gt

def distanceBB(box1, box2):

    eucl = np.linalg.norm(box1.center - box2.center)
    angl = Quaternion.distance(
        Quaternion(matrix=box1.rotation_matrix),
        Quaternion(matrix=box2.rotation_matrix))
    return eucl + angl


def generate_boxes(box, search_space=[[0, 0, 0]]):
    # Geenrate more candidate boxes based on prior and search space
    # Input : Prior position, search space and seaarch size
    # Output : List of boxes

    candidate_boxes = [getOffsetBB(box, offset) for offset in search_space]
    return candidate_boxes


def getDataframeGT(anno):
    df = {
        "scene": anno["scene"],
        "frame": anno["frame"],
        "track_id": anno["track_id"],
        "type": anno["type"],
        "truncated": anno["truncated"],
        "occluded": anno["occluded"],
        "alpha": anno["alpha"],
        "bbox_left": anno["bbox_left"],
        "bbox_top": anno["bbox_top"],
        "bbox_right": anno["bbox_right"],
        "bbox_bottom": anno["bbox_bottom"],
        "height": anno["height"],
        "width": anno["width"],
        "length": anno["length"],
        "x": anno["x"],
        "y": anno["y"],
        "z": anno["z"],
        "rotation_y": anno["rotation_y"]
    }
    return df


def getDataframe(anno, box, score):
    myquat = (box.orientation * Quaternion(axis=[1, 0, 0], radians=-np.pi / 2))
    df = {
        "scene": anno["scene"],
        "frame": anno["frame"],
        "track_id": anno["track_id"],
        "type": anno["type"],
        "truncated": anno["truncated"],
        "occluded": anno["occluded"],
        "alpha": 0.0,
        "bbox_left": 0.0,
        "bbox_top": 0.0,
        "bbox_right": 0.0,
        "bbox_bottom": 0.0,
        "height": box.wlh[2],
        "width": box.wlh[0],
        "length": box.wlh[1],
        "x": box.center[0],
        "y": box.center[1] + box.wlh[2] / 2,
        "z": box.center[2],
        "rotation_y":
        np.sign(myquat.axis[1]) * myquat.radians,  # this_anno["rotation_y"], #
        "score": score
    }
    return df


def saveTrackingResults(df_3D, dataset_loader, export=None, epoch=-1):

    for i_scene, scene in enumerate(df_3D.scene.unique()):
        new_df_3D = df_3D[df_3D["scene"] == scene]
        new_df_3D = new_df_3D.drop(["scene"], axis=1)
        new_df_3D = new_df_3D.sort_values(by=['frame', 'track_id'])

        os.makedirs(os.path.join("results", export, "data"), exist_ok=True)
        if epoch == -1:
            path = os.path.join("results", export, "data", "{}.txt".format(scene))
        else:
            path = os.path.join("results", export, "data",
                                "{}_epoch{}.txt".format(scene,epoch))

        new_df_3D.to_csv(
            path, sep=" ", header=False, index=False, float_format='%.6f')
