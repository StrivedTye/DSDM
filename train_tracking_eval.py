import argparse
import os
import random
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from Dataset import SiameseTrain, SiameseTest
from pointnet2.models.progressive_tracking import ProgressiveTrack
from loss_utils import SigmoidFocalClassificationLoss, WeightedSmoothL1Loss
from kitty_utils import lidar2bbox
from metrics import estimateOverlap, estimateAccuracy, Success, Precision
from test_tracking import test

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
parser.add_argument('--workers', type=int, default=6, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=2, help='# GPUs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--input_feature_num', type=int, default=0, help='number of input point features')
parser.add_argument('--roi_voxels', type=int, default=6, help='number of voxel each axis')
parser.add_argument('--data_dir', type=str,
                    default='/workspace/data/tracking/train_datasets/KITTI_Tracking/truly_using/',
                    help='dataset path')
parser.add_argument('--category_name', type=str, default='Car', help='Object to Track (Car/Pedestrian/Van/Cyclist)')
parser.add_argument('--save_root_dir', type=str, default='results', help='output folder')
parser.add_argument('--model', type=str, default='', help='model name for training resume')
parser.add_argument('--optimizer', type=str, default='', help='optimizer name for training resume')
parser.add_argument('--dataset', type=str, default='KITTI', help='which dataset for training (KITTI/PANDA)')

opt = parser.parse_args()
print(opt)

# torch.cuda.set_device(opt.main_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,3'

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = opt.save_root_dir

try:
    os.makedirs(save_dir)
except OSError:
    pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                    filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

# 1. Load data
train_data = SiameseTrain(
    input_size=1024,
    path=opt.data_dir,
    split='Train',
    category_name=opt.category_name,
    offset_BB=0,
    scale_BB=1.25, dataset=opt.dataset)

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    pin_memory=True)

test_data = SiameseTest(
    input_size=1024,
    path=opt.data_dir,
    split='TEST',
    category_name=opt.category_name,
    offset_BB=0,
    scale_BB=1.25, dataset=opt.dataset)

test_loader = torch.utils.data.DataLoader(
    test_data,
    collate_fn=lambda x: x,
    batch_size=1,
    shuffle=False,
    num_workers=int(opt.workers),
    pin_memory=True)

print('#Train data:', len(train_data), '#Test data:', len(test_data))
print(opt)

# 2. Define model, loss and optimizer
netR = ProgressiveTrack(input_channels=opt.input_feature_num, use_xyz=True, roi_voxels=opt.roi_voxels)
if opt.ngpu > 1:
    netR = torch.nn.DataParallel(netR, range(opt.ngpu))
if opt.model != '':
    netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))

netR.cuda()
# print(netR)

# Loss
criterion_seg = SigmoidFocalClassificationLoss()
criterion_similarity = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]), reduction='none').cuda()  # nn.MSELoss()
criterion_reg = WeightedSmoothL1Loss(code_weights=[1, 1, 1, 1])
criterion_prop_reg = WeightedSmoothL1Loss(code_weights=[1, 1, 1, 1])
criterion_prop_reg_vote = WeightedSmoothL1Loss(code_weights=[1, 1, 1])

# Optimizer
optimizer = optim.Adam(netR.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-06, weight_decay=1e-06)
if opt.optimizer != '':
    optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

mu_sim, mu_reg, mu_seg, mu_reg_proposal = 1.5, 1, 0.2, 0.2
# mu_sim, mu_reg, mu_seg, mu_reg_proposal = 1, 1, 1, 1
# mu_sim, mu_reg, mu_seg, mu_reg_proposal = 1, 1, 0.2, 0.2
# mu_sim, mu_reg, mu_seg, mu_reg_proposal = 2, 1, 0.2, 0.2
# mu_sim, mu_reg, mu_seg, mu_reg_proposal = 1.5, 1, 0.1, 0.1


# 3. Training and testing
for epoch in range(0, opt.nepoch):
    # scheduler.step(epoch)
    print('> Training set')
    print('>> Online epoch: #%d, lr=%f' % (epoch, scheduler.get_lr()[0]))
    # 3.1 switch to train mode
    torch.cuda.synchronize()
    netR.train()
    train_mse = 0.0
    timer = time.time()
    success_train = Success()
    precision_train = Precision()

    batch_similarity_loss = 0.0
    batch_reg_loss = 0.0
    batch_seg_t_loss = 0.0
    batch_seg_s_loss = 0.0
    batch_reg_propo_loss = 0.0
    for i, data in enumerate(tqdm(train_dataloader, 0, ncols=120)):
        if len(data[0]) == 1:
            continue
        torch.cuda.synchronize()

        # 3.1.1 load inputs and targets
        search_PC, search_origin_state, search_gt_state, label_search_semantic, \
        t_PC, t_origin_state, label_t_semantic = data

        search_origin_state = Variable(search_origin_state, requires_grad=False).cuda()
        # search_origin_state = search_origin_state.unsqueeze(1) #[B, m, 7]

        search_gt_state = Variable(search_gt_state, requires_grad=False).cuda()
        search_gt_state = search_gt_state.unsqueeze(1)  # [B, 1, 7]

        label_search_semantic = Variable(label_search_semantic, requires_grad=False).cuda()  # [B, N, 1]
        label_search_semantic = label_search_semantic.unsqueeze(-1)

        search_PC = Variable(search_PC, requires_grad=False).cuda()

        t_PC = Variable(t_PC, requires_grad=False).cuda()

        label_t_semantic = Variable(label_t_semantic, requires_grad=False).cuda()
        weight_semantic = torch.ones_like(label_t_semantic).cuda()
        label_t_semantic = label_t_semantic.unsqueeze(-1)  # [B, N, 1]

        t_origin_state = Variable(t_origin_state, requires_grad=False).cuda()
        t_origin_state = t_origin_state.unsqueeze(1)

        # 3.1.2 compute output
        optimizer.zero_grad()
        search_box_all, delta_p_all, score_all, t_score, search_score, \
        vote_xyz, pred_box, pred_centers = netR(search_PC, t_PC, search_origin_state, t_origin_state, search_gt_state)

        num_seeds = pred_box.shape[1]

        # generate label of proposal
        label_proposal_loc = search_gt_state.repeat(1, num_seeds, 1)
        label_proposal_loc = label_proposal_loc[:, :, [0, 1, 2, 6]]  # [B, 64, 4]

        label_proposal_loc_vote = search_gt_state.repeat(1, vote_xyz.shape[1], 1)
        label_proposal_loc_vote = label_proposal_loc_vote[:, :, [0, 1, 2]]  # [B, N, 3]

        # generate ground-truth delta_p for each step
        search_gt_state = search_gt_state.repeat(1, search_box_all.shape[1], 1)  # [B, m*4, 7]
        label_step_delta = search_gt_state - search_box_all  # [B, m*4, 7]
        label_delta_p = label_step_delta[:, 0:-num_seeds, [0, 1, 2, 6]]  # [B, m*3, 3]

        # generate similarity score labels
        sigma = 1
        num_step = int(label_step_delta.shape[1] / num_seeds)
        label_delta_p_score = label_step_delta[:, :, [0, 1, 6]].detach()  # [B, m*4, 3]
        label_delta_p_score[:, :, -1] = label_delta_p_score[:, :, -1] * 180 / np.pi
        label_delta_p_score = torch.flatten(label_delta_p_score, 0, 1)  # [B*m*4, 3]
        w_delta_p = torch.mul(label_delta_p_score, torch.Tensor([1, 1, 1 / 5.0]).cuda())
        dist = torch.norm(w_delta_p, dim=1, keepdim=True)
        label_score = torch.exp(-0.5 * dist / (sigma * sigma)).view(-1, num_seeds * num_step, 1)  # [B, m*4, 1]objectness_label_step

        # filter out proposals that far away from gt
        dist = torch.sum((pred_centers - label_proposal_loc[:, 0:num_seeds, 0:3]) ** 2, dim=-1)
        dist = torch.sqrt(dist + 1e-6)
        objectness_label = torch.zeros_like(dist).cuda()
        objectness_mask = torch.zeros_like(dist).cuda()
        objectness_label[dist < 0.3] = 1
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1
        objectness_label_step = objectness_label.repeat(1, num_step)
        objectness_mask_step = objectness_mask.repeat(1, num_step)

        # objectness_label_filter = objectness_label.clone()
        # objectness_label_filter[dist<0.6] = 1
        # objectness_label_filter = objectness_label_filter.repeat(1, num_step) # for step reg

        # similarty loss
        # loss_similarity = criterion_seg(score_all, objectness_label_step.unsqueeze(-1), torch.ones_like(score_all).cuda()).squeeze()
        loss_similarity = criterion_similarity(score_all.squeeze(-1), objectness_label_step)
        loss_similarity = torch.sum(loss_similarity * objectness_mask_step) \
                          / (torch.sum(objectness_mask_step) + 1e-6)

        # regression loss of progressive procedure
        loss_reg = criterion_reg(delta_p_all,
                                 label_delta_p,
                                 objectness_label_step[:, 0:-num_seeds])
        loss_reg = loss_reg.mean()

        # reg loss of proposals
        loss_proposal_reg = criterion_prop_reg(pred_box,
                                               label_proposal_loc,
                                               objectness_label)
        loss_proposal_reg_vote = criterion_prop_reg_vote(vote_xyz,
                                                         label_proposal_loc_vote,
                                                         label_search_semantic[:, 0:128, 0])

        loss_proposal_reg = loss_proposal_reg.mean() #loss_proposal_reg_vote.mean() #  +

        # classification loss: fore/back-ground
        loss_seg_t = criterion_seg(t_score, label_t_semantic, weight_semantic)  # [B, N, 1]
        loss_seg_t = loss_seg_t.mean()

        label_search_semantic = label_search_semantic[:, 0:128, 0:] #torch.gather(label_search_semantic, 1, search_seed_inds.long().unsqueeze(-1)) #
        loss_seg_s = criterion_seg(search_score, label_search_semantic, weight_semantic[:,0:128])
        loss_seg_s = loss_seg_s.mean()

        # total loss
        loss = mu_sim * loss_similarity \
               + mu_reg * loss_reg \
               + mu_seg * loss_seg_s \
               + mu_reg_proposal * loss_proposal_reg

        # 3.1.3 compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # 3.1.4 update training error
        idx_box = torch.argmax(score_all, dim=1, keepdim=True)  # [B, 1, 1]
        selected_box = torch.gather(search_box_all, 1, idx_box.repeat(1, 1, search_box_all.shape[2]))  # [B, 1, 7]
        selected_box_cpu = selected_box.squeeze(1).detach().cpu().numpy()  # [B, 7]
        selected_bbox = lidar2bbox(selected_box_cpu)

        gt_state_cpu = search_gt_state[:, 0, :].detach().cpu().numpy()  # [B, 7]
        gt_bbox = lidar2bbox(gt_state_cpu)

        for pred, gt in zip(selected_bbox, gt_bbox):
            this_overlap = estimateOverlap(gt, pred, dim=3)
            this_accuracy = estimateAccuracy(gt, pred, dim=3)
            success_train.add_overlap(this_overlap)
            precision_train.add_accuracy(this_accuracy)

        train_mse = train_mse + loss.data * t_PC.shape[0]
        batch_similarity_loss += loss_similarity.data
        batch_reg_loss += loss_reg.data
        batch_seg_t_loss += loss_seg_t.data
        batch_seg_s_loss += loss_seg_s.data
        batch_reg_propo_loss += loss_proposal_reg.data
        if (i + 1) % 20 == 0:
            print('\n>>> batch: %03d ----' % (i + 1))
            print('>>>> similar_loss: %f, reg_loss: %f, seg_templ_loss: %f, seg_sear_loss: %f, reg_propo_loss: %f'
                  % (batch_similarity_loss / 20,
                     batch_reg_loss / 20,
                     batch_seg_t_loss / 20,
                     batch_seg_s_loss / 20,
                     batch_reg_propo_loss / 20)
                  )

            batch_similarity_loss = 0.0
            batch_reg_loss = 0.0
            batch_seg_t_loss = 0.0
            batch_seg_s_loss = 0.0
            batch_reg_propo_loss = 0.0

    print('>> train success: %f, train precision: %f' % (success_train.average, precision_train.average))
    scheduler.step()


    # time taken
    train_mse = train_mse / len(train_data)
    torch.cuda.synchronize()
    timer = time.time() - timer
    timer = timer / len(train_data)
    print('>> time to learn 1 sample = %f (ms)' % (timer * 1000))

    torch.save(netR.state_dict(), '%s/vote_netR_%d.pth' % (save_dir, epoch))
    torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))

    # 3.2 switch to evaluate mode
    if epoch<10: continue
    torch.cuda.synchronize()
    netR.eval()
    with torch.no_grad():
        success_test, precision_test, A, R = test(test_loader,
                                                  netR,
                                                  epoch=1,
                                                  shape_aggregation='firstandprevious',
                                                  reference_BB='previous_result',
                                                  IoU_Space=3, db=opt.dataset)

    test_similarity_loss = 0.0
    test_reg_loss = 0.0
    test_seg_t_loss = 0.0
    test_seg_s_loss = 0.0
    test_reg_propo_loss = 0.0
    timer = time.time()

    print('>> validation success: %f, validation precision: %f' % (success_test, precision_test))

    # time taken
    torch.cuda.synchronize()
    timer = time.time() - timer
    timer = timer / len(test_data)
    print('>> time to learn 1 sample = %f (ms)' % (timer * 1000))

    # log
    logging.info('Epoch#%d: '
                 'train error=%e, '
                 'test S/P=%e, %e, %e, %e '
                 'lr = %f'
                 % (epoch,
                    train_mse,
                    success_test, precision_test, A, R,
                    scheduler.get_lr()[0]))
