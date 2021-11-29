import time
import os
import logging
import argparse
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from datetime import datetime
from pyquaternion import Quaternion

from metrics import AverageMeter, Success, Precision, Robustness
from metrics import estimateOverlap, estimateAccuracy
from searchspace import KalmanFiltering, ParticleFiltering, GaussianMixtureModel
from Dataset import SiameseTest, waymoDataset
import kitty_utils as utils
from pointnet2.models.progressive_tracking import ProgressiveTrack

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vis(PC, boxs, gt_box):

    ColorNames = ['#4682b4',  # stealblue
                  '#FF8C00',  # darkorange
                  '#808000',  # olive
                  'black',
                  '#8FBC8F',  # darkseagreen
                  'red',
                  ]

    # Create figure for TRACKING
    fig = plt.figure(1, figsize=(9, 6), facecolor="white")
    plt.rcParams['savefig.dpi'] = 320
    plt.rcParams['figure.dpi'] = 320

    # Create axis in 3D
    ax = fig.gca(projection='3d')

    # Scatter plot the cropped point cloud
    ratio = 1
    ax.scatter(
        PC.points[0, ::ratio],
        PC.points[1, ::ratio],
        PC.points[2, ::ratio] / 2 - 1,
        s=3,
        c=PC.points[2, ::ratio])

    # point order to draw a full Box
    order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]

    # Plot GT box
    ax.plot(
        gt_box.corners()[0, order],
        gt_box.corners()[1, order],
        gt_box.corners()[2, order] / 2 - 1,
        color='red',
        alpha=0.5,
        linewidth=5,
        linestyle="-")

    for id, box in enumerate(boxs):
        box = box.corners()

        ax.plot(
            box[0, order],
            box[1, order],
            box[2, order] / 2 - 1,
            color=ColorNames[id],
            alpha=1,
            linewidth=2,
            linestyle="-")
    # ax.view_init(90, 0)
    ax.view_init(30, 60)
    plt.axis('off')
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.clf()

def test(loader,model,epoch=-1,shape_aggregation="",reference_BB="",max_iter=-1,IoU_Space=3, db='KITTI'):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    Success_main = Success()
    Precision_main = Precision()
    Success_batch = Success()
    Precision_batch = Precision()

    Robustness_batch = Robustness()
    Robustness_main = AverageMeter()

    Success_range = [Success(), Success(), Success()]
    Precision_range = [Precision(), Precision(), Precision()]

    Success_class = [Success(), Success(), Success(), Success()]
    Precision_class = [Precision(), Precision(), Precision(), Precision()]

    # switch to evaluate mode
    model.eval()
    end = time.time()

    dataset = loader.dataset
    batch_num = 0
    num_seeds = 63
    search_space_sampler = KalmanFiltering([0.5, 0.5, 10])

    with tqdm(enumerate(loader), total=len(dataset.list_of_anno), ncols=120) as t:
        for batch in loader:          
            batch_num = batch_num+1
            # measure data loading time
            data_time.update((time.time() - end))
            for PCs, BBs, list_of_anno in batch: # tracklet
                results_BBs = []
                results_normalized_BBs = []
                search_space_sampler.reset()

                for i, _ in enumerate(PCs): # frame
                    this_anno = list_of_anno[i]
                    this_BB = BBs[i]
                    this_PC = PCs[i]

                    # INITIAL FRAME
                    if i == 0:
                        box = BBs[i]
                        results_BBs.append(box)
                        if not model.module.learn_R:

                            model_PC = utils.getModel([PCs[0]],
                                                      [results_BBs[0]],
                                                      offset=dataset.offset_BB,
                                                      scale=dataset.scale_BB)
                            t_origin_state = np.array([0, 0, 0,
                                                       results_BBs[0].wlh[1],
                                                       results_BBs[0].wlh[0],
                                                       results_BBs[0].wlh[2],
                                                       0])  # [x, y, z, l, w, h, theta]
                            t_origin_state_torch = torch.from_numpy(t_origin_state).float()
                            t_origin_state_torch = t_origin_state_torch.unsqueeze(0).unsqueeze(0)
                            t_origin_state_torch = Variable(t_origin_state_torch, requires_grad=False).cuda()

                            model_PC_torch = utils.regularizePC(model_PC, dataset.input_size,istrain=False).unsqueeze(0)
                            model_PC_torch = Variable(model_PC_torch, requires_grad=False).cuda()


                            candidate_PC, origin_state, gt_state, _ = utils.cropAndCenterPC_label(this_PC,
                                                                                                  this_BB,
                                                                                                  this_BB,
                                                                                                  [0, 0, 0],
                                                                                                  offset=dataset.offset_BB,
                                                                                                  scale=dataset.scale_BB)

                            candidate_PCs = utils.regularizePC(candidate_PC, dataset.input_size, istrain=False)
                            candidate_PCs_torch = candidate_PCs.unsqueeze(0)
                            candidate_PCs_torch = Variable(candidate_PCs_torch, requires_grad=False).cuda()

                            # proposals for initializing
                            search_space = np.random.uniform(low=-0.3, high=0.3, size=[num_seeds, 4])
                            search_space[:, -2] = search_space[:, -2] * 0.5
                            # search_space[:, -1] = search_space[:, -1] * 5.0
                            parallel_seeds = np.reshape(origin_state, (1, 7))
                            parallel_seeds = np.repeat(parallel_seeds, num_seeds + 1, 0)  # [64+1, 7]
                            parallel_seeds[1:, [0, 1, 2, 6]] = parallel_seeds[1:, [0, 1, 2, 6]] + search_space
                            c_origin_state_torch = torch.from_numpy(parallel_seeds).float()
                            c_origin_state_torch = c_origin_state_torch.unsqueeze(0)  # [1, 65, 7]
                            c_origin_state_torch = Variable(c_origin_state_torch, requires_grad=False).cuda()

                            search_box_all, \
                            delta_p_all, \
                            score_all, \
                            t_score, \
                            search_score, _, _, _ = model(candidate_PCs_torch,
                                                          model_PC_torch,
                                                          c_origin_state_torch,
                                                          t_origin_state_torch,
                                                          t_origin_state_torch, False)
                    else:
                        previous_BB = BBs[i - 1]

                        # DEFINE REFERENCE BB
                        if ("previous_result".upper() in reference_BB.upper()):
                            ref_BB = results_BBs[-1]
                        elif ("previous_gt".upper() in reference_BB.upper()):
                            ref_BB = previous_BB
                            # ref_BB = utils.getOffsetBB(this_BB,np.array([-1,1,1]))
                        elif ("current_gt".upper() in reference_BB.upper()):
                            ref_BB = this_BB

                        new_ref_BB = utils.Centerbox(ref_BB, this_BB)
                        degree = new_ref_BB.orientation.yaw_pitch_roll[0] /np.pi*180
                        offset_true = [new_ref_BB.center[0],
                                       new_ref_BB.center[1],
                                       degree]
                        # search_space = np.array([offset_true])
                        # print('\n',offset_true)
                        search_space = search_space_sampler.sample(num_seeds)
                        # search_space = np.concatenate([np.array([offset_true]), search_space], 0)

                        candidate_PC, origin_state, gt_state, _ = utils.cropAndCenterPC_label(this_PC,
                                                                                              ref_BB,
                                                                                              this_BB,
                                                                                              offset_true,
                                                                                              offset=dataset.offset_BB,
                                                                                              scale=dataset.scale_BB)
                        gt_state[-1] = gt_state[-1] / np.pi * 180

                        candidate_PCs = utils.regularizePC(candidate_PC, dataset.input_size, istrain=False)

                        # np.set_printoptions(precision=3, suppress=True)
                        # print('\ngt{}:{}'.format(i, gt_state))
                        # print('origin{}:{}'.format(i, origin_state))

                        candidate_PCs_torch = candidate_PCs.unsqueeze(0)
                        candidate_PCs_torch = Variable(candidate_PCs_torch, requires_grad=False).cuda()

                        parallel_seeds = np.reshape(origin_state, (1, 7))
                        parallel_seeds = np.repeat(parallel_seeds, num_seeds + 1, 0)  # [64+1, 7]
                        parallel_seeds[1:, [0, 1]] = parallel_seeds[1:, [0, 1]] + search_space[:, 0:2]
                        parallel_seeds[1:, 6] = parallel_seeds[1:, 6] + (search_space[:, -1] / 180 * np.pi)


                        c_origin_state_torch = torch.from_numpy(parallel_seeds).float()
                        c_origin_state_torch = c_origin_state_torch.unsqueeze(0) #[1, 65, 7]
                        c_origin_state_torch = Variable(c_origin_state_torch, requires_grad=False).cuda()

                        # AGGREGATION: IO vs ONLY0 vs ONLYI vs ALL
                        if ("firstandprevious".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[0], PCs[i-1]],
                                                      [results_BBs[0],results_BBs[i-1]],
                                                      offset=dataset.offset_BB,
                                                      scale=dataset.scale_BB)
                        elif ("first".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[0]],
                                                      [results_BBs[0]],
                                                      offset=dataset.offset_BB,
                                                      scale=dataset.scale_BB)
                        elif ("previous".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel([PCs[i-1]],
                                                      [results_BBs[i-1]],
                                                      offset=dataset.offset_BB,
                                                      scale=dataset.scale_BB)
                        elif ("all".upper() in shape_aggregation.upper()):
                            model_PC = utils.getModel(PCs[:i],
                                                      results_BBs,
                                                      offset=dataset.offset_BB,
                                                      scale=dataset.scale_BB)
                        else:
                            model_PC = utils.getModel(PCs[:i],
                                                      results_BBs,
                                                      offset=dataset.offset_BB,
                                                      scale=dataset.scale_BB)

                        t_origin_state = np.array([0, 0, 0,
                                                   results_BBs[0].wlh[1],
                                                   results_BBs[0].wlh[0],
                                                   results_BBs[0].wlh[2],
                                                   0])  # [x, y, z, l, w, h, theta]
                        t_origin_state_torch = torch.from_numpy(t_origin_state).float()
                        t_origin_state_torch = t_origin_state_torch.unsqueeze(0).unsqueeze(0)
                        t_origin_state_torch = Variable(t_origin_state_torch, requires_grad=False).cuda()

                        model_PC_torch = utils.regularizePC(model_PC, int(dataset.input_size), istrain=False).unsqueeze(0)
                        model_PC_torch = Variable(model_PC_torch, requires_grad=False).cuda()

                        search_box_all, \
                        delta_p_all, \
                        score_all, \
                        t_score, \
                        search_score, _, _, _ = model(candidate_PCs_torch,
                                                      model_PC_torch,
                                                      c_origin_state_torch,
                                                      t_origin_state_torch, None, False)

                        #***********************select the best one from all steps************
                        score_all = score_all.sigmoid()
                        score_all_ = score_all.view(-1, num_seeds+1, 1)#[4, m, 1]
                        score_all_batch = torch.max(score_all_, dim=0)[0] #[m, 1]
                        score_all_batch = score_all_batch.detach().cpu().numpy()

                        search_box_all = search_box_all.view(-1, 7)
                        search_box_all[:, -1]=search_box_all[:, -1] / np.pi *180

                        w_delta_p = torch.mul(search_box_all[:,[0,1]], torch.Tensor([1, 1]).cuda())
                        dist = torch.norm(w_delta_p, dim=1, keepdim=True)
                        score_dist = torch.exp(-(dist ** 2) / 0.8) # [B*4, 1]
                        # print('\ndist:', score_dist)
                        # print('\nscore:', score_all)
                        score_all = score_all.view(-1, 1)#*score_dist
                        # print('\nfinal:', score_all)
                        idx_box = torch.argmax(score_all, dim=0, keepdim=True)  # [1, 1]

                        #***********************select the best one from the last step************
                        # score_all_ = score_all.view(-1, num_seeds+1, 1)#[4, m, 1]
                        # score_all_batch = score_all_[3, :, :] #[m, 1]
                        # idx_box = torch.argmax(score_all_batch, dim=0, keepdim=True)  # [1, 1]
                        # score_all_batch = score_all_batch.detach().cpu().numpy()
                        #
                        # search_box_all = search_box_all.view(-1, num_seeds+1, 7)
                        # search_box_all = search_box_all[3, :, :]
                        # search_box_all[:, -1]=search_box_all[:, -1] / np.pi *180

                        #**********************selection*****************
                        selected_box = torch.gather(search_box_all, 0,
                                                    idx_box.repeat(1, search_box_all.shape[-1]))
                        selected_box_cpu = selected_box.detach().cpu().numpy()  # [1, 7]
                        selected_box_cpu = selected_box_cpu[0, [0, 1, 2, 6]]
                        box = utils.getOffsetBB(ref_BB, selected_box_cpu)
                        # box = utils.getOffsetBB(ref_BB, gt_state[[0,1,2,6]])

                        results_BBs.append(box)
                        search_space_sampler.addData(search_space, score_all_batch[1:,0])

                        #**********************Visualization*****************
                        # search_box_all = search_box_all.detach().cpu().numpy()
                        # vis_boxs = utils.generate_boxes(ref_BB, search_box_all[:, [0,1,2,6]])
                        # vis_gt_box = utils.getOffsetBB(ref_BB, gt_state[[0,1,2,6]])
                        # vis_boxs = [utils.Centerbox(box, this_BB) for box in vis_boxs] # this_BB for saving normalized results
                        # vis_gt_box = utils.Centerbox(vis_gt_box, ref_BB)
                        # vis(candidate_PC, vis_boxs, vis_gt_box)

                        # For saving box of each step
                        # for b in vis_boxs:
                        #     results_normalized_BBs.append(b.corners().flatten())

                    #######################For saving normalized results##############
                    # normalized_box = utils.Centerbox(box, this_BB)
                    # results_normalized_BBs.append(normalized_box.corners().flatten())


                    # estimate overlap/accuracy fro current sample
                    if db == 'PANDA' or db == 'WAYMO':
                        box_overlap1 = copy.deepcopy(BBs[i])
                        box_overlap2 = copy.deepcopy(results_BBs[-1])
                        box_overlap1.rotate(Quaternion(axis=[1, 0, 0], angle=np.pi / 2))
                        box_overlap2.rotate(Quaternion(axis=[1, 0, 0], angle=np.pi / 2))
                        this_overlap = estimateOverlap(box_overlap1, box_overlap2, dim=IoU_Space)
                        this_accuracy = estimateAccuracy(box_overlap1, box_overlap2, dim=IoU_Space)
                    elif db == 'KITTI':
                        this_overlap = estimateOverlap(BBs[i], results_BBs[-1], dim=IoU_Space)
                        this_accuracy = estimateAccuracy(BBs[i], results_BBs[-1], dim=IoU_Space)


                    Success_main.add_overlap(this_overlap)
                    Precision_main.add_accuracy(this_accuracy)
                    Success_batch.add_overlap(this_overlap)
                    Precision_batch.add_accuracy(this_accuracy)
                    Robustness_batch.add_overlap(this_overlap)

                    range = np.linalg.norm(this_BB.center)
                    if range < 30:
                        Success_range[0].add_overlap(this_overlap)
                        Precision_range[0].add_accuracy(this_accuracy)
                    elif range < 60:
                        Success_range[1].add_overlap(this_overlap)
                        Precision_range[1].add_accuracy(this_accuracy)
                    else:
                        Success_range[2].add_overlap(this_overlap)
                        Precision_range[2].add_accuracy(this_accuracy)

                    if db == 'PANDA' or db == 'KITTI':
                        if this_anno["type"] == "Car":
                            Success_class[0].add_overlap(this_overlap)
                            Precision_class[0].add_accuracy(this_accuracy)
                        elif this_anno["type"] == "Pedestrian":
                            Success_class[1].add_overlap(this_overlap)
                            Precision_class[1].add_accuracy(this_accuracy)
                        elif this_anno["type"] == "Bicycle" or this_anno["type"] == "Cyclist":
                            Success_class[2].add_overlap(this_overlap)
                            Precision_class[2].add_accuracy(this_accuracy)
                        else:
                            Success_class[3].add_overlap(this_overlap)
                            Precision_class[3].add_accuracy(this_accuracy)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    t.update(1)

                    if Success_main.count >= max_iter and max_iter >= 0:
                        return Success_main.average, Precision_main.average


                t.set_description('Test {}: '.format(epoch)+
                                  'Time {:.3f}s '.format(batch_time.avg)+
                                  '(it:{:.3f}s) '.format(batch_time.val)+
                                  'Data:{:.3f}s '.format(data_time.avg)+
                                  '(it:{:.3f}s), '.format(data_time.val)+
                                  'Succ/Prec:'+
                                  '{:.2f}/'.format(Success_main.average)+
                                  '{:.2f}'.format(Precision_main.average))

                logging.info('batch {} '.format(batch_num)+'Succ/Prec:'+
                                  '{:.1f}/'.format(Success_batch.average)+
                                  '{:.1f}'.format(Precision_batch.average))
                Robustness_main.update(Robustness_batch.average, Robustness_batch.count)
                Success_batch.reset()
                Precision_batch.reset()
                Robustness_batch.reset()

                ##############################Save normalized results to txt
                # tracker = "DSDM_all_step_gaussian"
                # os.makedirs(os.path.join("./visual_results", f"{batch_num:04.0f}", "Tracking"), exist_ok=True)
                # file_name = os.path.join("./visual_results", f"{batch_num:04.0f}", "Tracking", f"{i}_{tracker}.txt")
                # np.savetxt(file_name, results_normalized_BBs, fmt='%f')
                #
                # file_name = os.path.join("./visual_results", f"{batch_num:04.0f}", "Tracking", "scene_tracklet_ID.txt")
                # scene_tracklet_ID = np.array([int(list_of_anno[0]["scene"]),
                #                           int(list_of_anno[0]["track_id"])])
                # np.savetxt(file_name, scene_tracklet_ID, fmt='%d')


    if True:
        logging.info("0-30m, mean Succ/Prec {}/{}".format(Success_range[0].average, Precision_range[0].average))
        logging.info("30-60m, mean Succ/Prec {}/{}".format(Success_range[1].average, Precision_range[1].average))
        logging.info(">60m, mean Succ/Prec {}/{}".format(Success_range[2].average, Precision_range[2].average))

        logging.info("Car, mean Succ/Prec {}/{}".format(Success_class[0].average, Precision_class[0].average))
        logging.info("Pedestrian, mean Succ/Prec {}/{}".format(Success_class[1].average, Precision_class[1].average))
        logging.info("Cyclist, mean Succ/Prec {}/{}".format(Success_class[2].average, Precision_class[2].average))
        logging.info("Other, mean Succ/Prec {}/{}".format(Success_class[3].average, Precision_class[3].average))

    return Success_main.average, Precision_main.average, np.mean(Success_main.overlaps), Robustness_main.avg #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs') 
    parser.add_argument('--save_root_dir', type=str, default='./model/car_model_panda/', help='output folder')
    parser.add_argument('--data_dir', type=str, default = '/workspace/data/tracking/train_datasets/PandaSet/',  help='dataset path')
    parser.add_argument('--model', type=str, default = 'vote_netR_46.pth', help='model name for training resume')
    parser.add_argument('--category_name', type=str, default='Car', help='Object to Track (Car/Pedestrian/Van/Cyclist)')
    parser.add_argument('--shape_aggregation',required=False,type=str,default="firstandprevious",help='Aggregation of shapes (first/previous/firstandprevious/all)')
    parser.add_argument('--reference_BB',required=False,type=str,default="previous_result",help='previous_result/previous_gt/current_gt')
    parser.add_argument('--model_fusion',required=False,type=str,default="pointcloud",help='early or late fusion (pointcloud/latent/space)')
    parser.add_argument('--IoU_Space',required=False,type=int,default=3,help='IoUBox vs IoUBEV (2 vs 3)')
    parser.add_argument('--dataset', type=str, default='PANDA', help='which dataset for training (KITTI/PANDA)')

    args = parser.parse_args()
    print(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                        filename=os.path.join(args.save_root_dir, datetime.now().strftime('%Y-%m-%d %H-%M-%S.log')),
                        level=logging.INFO)
    logging.info('======================================================')

    args.manualSeed = 1
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    netR = ProgressiveTrack(input_channels=0, use_xyz=True)

    netR = torch.nn.DataParallel(netR, range(args.ngpu))
    # print(netR)

    #+++++++++++++++++++++++++++++GFLOPs and Parameters++++++++++++++++++++++++++++#
    # from thop import profile
    # from pointnet2.utils.pointnet2_utils import QueryAndGroup
    # def countQG(m, x, y):
    #     n1 = x[0].size(1)
    #     n2 = x[1].size(1)
    #     # sum + sum + matmul + element-wise add + ew muliply by 2 + fast sort
    #     total_ops = 5 * n1 + 5 * n2 + 5 * n1 * n2 + 4 * n1 * n2 + n1 + n2
    #     m.total_ops = torch.Tensor([int(total_ops)])
    #
    # t = torch.randn(1, 2048, 3).cuda()
    # s = torch.randn(1, 2048, 3).cuda()
    # s_b = torch.randn(1, 32, 7).cuda()
    # t_b = torch.randn(1, 1, 7).cuda()
    # flops, params = profile(netR, inputs=(t,s, s_b, t_b), custom_ops={QueryAndGroup: countQG})
    # total_num = sum(p.numel() for p in netR.parameters())
    # print(flops,params)
    # print('total num:', total_num)
    # exit()
    #+++++++++++++++++++++++++++++GFLOPs and Parameters++++++++++++++++++++++++++++#

    if args.model != '':
        model_path = os.path.join(args.save_root_dir, args.model)
        netR.load_state_dict(torch.load(model_path))
    netR.cuda()
    torch.cuda.synchronize()

    # Car/Pedestrian/Van/Cyclist
    dataset_Test = SiameseTest(
            input_size=1024,
            path= args.data_dir,
            split='Test',
            category_name=args.category_name,
            offset_BB=0,
            scale_BB=1.25, dataset=args.dataset)
    # dataset_path = '/workspace/data/tracking/test_datasets/waymo_valid_extract/'
    # benchmark_dir = dataset_path + 'benchmark'
    # dataset_Test = waymoDataset(benchmark_dir, dataset_path, input_size=1024)


    test_loader = torch.utils.data.DataLoader(
        dataset_Test,
        collate_fn=lambda x: x,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True)

    Success_run = AverageMeter()
    Precision_run = AverageMeter()

    if dataset_Test.isTiny():
        max_epoch = 2
    else:
        max_epoch = 1
    for epoch in range(max_epoch):
        with torch.no_grad():
            Succ, Prec, A, R = test(
                test_loader,
                netR,
                epoch=epoch + 1,
                shape_aggregation=args.shape_aggregation,
                reference_BB=args.reference_BB,
                IoU_Space=args.IoU_Space, db=args.dataset)
        Success_run.update(Succ)
        Precision_run.update(Prec)
        logging.info("mean Succ/Prec {}/{}".format(Success_run.avg,Precision_run.avg))
        logging.info("mean Accu/Robu {}/{}".format(A,R))
