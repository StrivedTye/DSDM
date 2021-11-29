from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from collections import namedtuple
import torch.nn.functional as F
from torch.autograd import Variable

from pointnet2.utils.pointnet2_modules import PointnetSAModule
from pointnet2.utils.linalg_utils import FarthestPointSample
from roiaware_pool3d.roiaware_pool3d_utils import RoIAwarePool3d, points_in_boxes_gpu
from iou3d_nms.iou3d_nms_utils import nms_normal_gpu, nms_gpu
from pointnet2.models.transformer import Transformer
from pointnet2.models.backbone import PointNet2Ind, PointNet2
import invmat

class ProgressiveTrack(nn.Module):

    def __init__(self, input_channels=3, use_xyz=True, roi_voxels=6, objective=False):
        super(ProgressiveTrack, self).__init__()

        self.num_R = 3

        self.backbone_net = PointNet2(input_channels, use_xyz)
        # self.backbone_net = PointNet2Ind(input_channels)

        # self.transformer = Transformer(emb_dims=256, n_blocks=1, dropout=0, ff_dims=512, n_heads=4)

        self.semantic_cls = (pt_utils.Seq(256)
                             .conv1d(128, bn=True)
                             .conv1d(128, bn=True)
                             .conv1d(1, activation=None))

        # self.proposal_layer = Proposal_Layer(num_proposal=64)
        self.proposal_layer = Proposal_Layer_vote(num_proposal=64)

        # self.canonical_xyz_mlp = (pt_utils.Seq(3)
        #                           .conv1d(64, bn=True)
        #                           .conv1d(64, activation=None))

        self.roiaware_pool3d_layer = RoIAwarePool3d(out_size=roi_voxels, max_pts_each_voxel=128)

        self.proposal_feat = (pt_utils.Seq(roi_voxels ** 3 * 256)
                              .conv1d(256, bn=True)
                              .dropout(0.3)
                              .conv1d(256, bn=True))

        self.similar_feat = (pt_utils.Seq(256).conv1d(256, bn=True))
        self.reg_feat = (pt_utils.Seq(256).conv1d(256, bn=True))

        self.learn_R = True
        if self.learn_R:
            self.R_module = nn.ModuleList()
            for i in range(self.num_R):
                self.R_module.append((pt_utils.Seq(256)
                                      .conv1d(256, bn=True)
                                      .conv1d(256, bn=True)
                                      .conv1d(3+1, activation=None)))
        else:
            self.R_module = nn.ParameterList([nn.Parameter(torch.zeros([256, 4]), requires_grad=False)
                                              for i in range(self.num_R)])
            self.first = True

        self.similar_pred = (pt_utils.Seq(512)
                             .conv1d(256, bn=True)
                             .conv1d(256, bn=True)
                             .conv1d(1, activation=None))

    def roi_aware_pool(self, coord, roi, feat):
        """
        :param coord: coordinate [B, N, 3]
        :param roi:  bounding box [B, m, 7]
        :param feat: point-wise feature [B, C, N]
        :return: pooled_feature [B*m, voxel_num, voxel_num, voxel_num, C]
        """

        feat = feat.transpose(1, 2).contiguous()

        batch_size = coord.shape[0]
        pooled_feat_list = []
        for i in range(batch_size):
            cur_coord = coord[i]
            cur_roi = roi[i].contiguous()
            cur_feat = feat[i]
            pooled_feat = self.roiaware_pool3d_layer(cur_roi, cur_coord, cur_feat, pool_method='max')  # [m, 6, 6, 6, C]
            pooled_feat_list.append(pooled_feat)
        pooled_features = torch.cat(pooled_feat_list, dim=0)  # [B*m, ...]
        return pooled_features  # [B*m, 6, 6, 6, C]

    def recursive_LS(self,  X, search_box, search_gt_state, R_old=None, P_old=None, first=True):

        X = X.view(search_box.size(0), search_box.size(1), -1) #[B, m, 256]
        search_gt_state = search_gt_state.repeat(1, search_box.size(1), 1)  # [B, m, 7]
        Y = search_gt_state - search_box  # [B, m, 7]
        Y = Y[:, :, [0, 1, 2, 6]] #[B, m, 4]

        index_for_RR = int(search_box.size(1)/2)
        X = X[:, 0:index_for_RR, :].contiguous().view(-1, X.size(-1)) #[n, 256]
        Y = Y[:, 0:index_for_RR, :].contiguous().view(-1, Y.size(-1)) #[n, 4]

        Xt = X.transpose(0, 1).contiguous()
        if first:
            P = Xt.mm(X) + torch.eye(X.size(-1)).to(X) * 0.1
            R = torch.inverse(P).mm(Xt).mm(Y)
        else:
            P_temp = torch.inverse(X.mm(P_old).mm(Xt)+ torch.eye(Xt.size(-1)).to(X))
            P = P_old - P_old.mm(Xt).mm(P_temp).mm(X).mm(P_old) #[256, 256]
            R = R_old + P.mm(Xt).mm(Y-X.mm(R_old))
        return R, P

    def cal_R_closed_form(self, X, search_box, search_gt_state, train=True):
        """
        (I+X'X)^(-1)X'Y ==> X'(I+XX')^(-1)Y
        X: fused_feat, i.e. C_reg_feat - T_reg_feat, of shape [B*m, 256, 1]
        search_gt_state: ground-truth of seach region [B, 1, 7]
        search_box: candidate boxes, of shape [B, m, 7]
        """

        X = X.view(search_box.size(0), search_box.size(1), -1) #[B, m, 256]
        search_gt_state = search_gt_state.repeat(1, search_box.size(1), 1)  # [B, m, 7]
        Y = search_gt_state - search_box  # [B, m, 7]
        Y = Y[:, :, [0, 1, 2, 6]] #[B, m, 4]

        index_for_RR = 4 #if train else 64 #int(search_box.size(1)/2)

        # X = X[:, 0:index_for_RR, :]
        # Y = Y[:, 0:index_for_RR, :]
        # Xt = X.transpose(1, 2).contiguous()
        # H = X.bmm(Xt) + torch.eye(index_for_RR).to(X) * 0.1
        # H_inv = torch.inverse(H) #[B, m, m]
        # pinv = Xt.bmm(H_inv) # [B, 256, m]
        # R = pinv.bmm(Y) # [B, 256, 4]


        X = X[:, 0:index_for_RR, :].contiguous().view(-1, X.size(-1)) #[n, 256]
        Y = Y[:, 0:index_for_RR, :].contiguous().view(-1, Y.size(-1)) #[n, 4]

        # (I+X'X)^(-1)X'T
        # Xt = X.transpose(0, 1).contiguous()
        # P = Xt.mm(X) + torch.eye(X.size(-1)).to(X) * 0.1
        # R = torch.inverse(P).mm(Xt).mm(Y)

        # X'(I+XX')^(-1)Y
        Xt = X.transpose(0, 1).contiguous()
        P = X.mm(Xt) + torch.eye(X.size(0)).to(X) * 0.1
        R = Xt.mm(torch.inverse(P)).mm(Y)

        return R

    def progressive(self,
                    idx, num_seed,
                    pooled_feat_f, box,
                    search_xyz, search_feat,
                    T_similar_feat, T_reg_feat, search_gt_state, learn_R=True, train=True):

        ## loop
        C_proposal_feat = self.proposal_feat(pooled_feat_f)  # [B*m, 256, 1]
        C_similar_feat = self.similar_feat(C_proposal_feat)
        C_reg_feat = self.reg_feat(C_proposal_feat)

        # calculating similarity
        # score = F.cosine_similarity(T_similar_feat,
        #                             C_similar_feat,
        #                             dim=1)  # [B*m, 1]
        # score = score.view(-1, num_seed, 1)

        fused_similar_feat = torch.cat((C_similar_feat, T_similar_feat), dim=1)
        # fused_similar_feat = C_similar_feat
        score = self.similar_pred(fused_similar_feat)
        score = score.view(-1, num_seed, 1)

        # regression
        # step 0->1
        fused_feat = C_reg_feat - T_reg_feat
        if not learn_R:
            if search_gt_state is not None:
                # if self.first:
                #     R, P = self.recursive_LS(fused_feat, box, search_gt_state, first=self.first)
                #     self.R_module[idx].copy_(R)
                #     self.P_module[idx].copy_(P)
                # else:
                #     R, P = self.recursive_LS(fused_feat, box, search_gt_state, self.R_module[idx], self.P_module[idx], first=self.first)
                #     self.R_module[idx].copy_(R)
                #     self.P_module[idx].copy_(P)
                # delta_p = fused_feat.squeeze(-1).matmul(R)
                # delta_p = delta_p.view(box.size(0), num_seed, -1)# [B, m, 4]

                R = self.cal_R_closed_form(fused_feat, box, search_gt_state, train)
                if not train: self.R_module[idx].copy_(R)

                # delta_p = fused_feat.squeeze(-1).view(box.size(0), num_seed, -1).bmm(R)  # [B, m, 4]
                delta_p = fused_feat.squeeze(-1).matmul(R)
                delta_p = delta_p.view(box.size(0), num_seed, -1)# [B, m, 4]
            else:
                # delta_p = fused_feat.squeeze(-1).view(box.size(0), num_seed, -1).bmm(self.R_module[idx]) #[B, m, 4]
                delta_p = fused_feat.squeeze(-1).matmul(self.R_module[idx])
                delta_p = delta_p.view(box.size(0), num_seed, -1)# [B, m, 4]


            search_pooled_feat_f, search_box = self.get_pooled_feat(delta_p, box, search_xyz, search_feat)
            return search_pooled_feat_f, search_box, score, delta_p

        else:
            delta_p = self.R_module[idx](fused_feat)  # [B*m, 4, 1]
            delta_p = delta_p.transpose(1, 2).contiguous()
            delta_p = delta_p.view(-1, num_seed, 1, 4).squeeze(2)  # [B, m, 4]

            search_pooled_feat_f, search_box = self.get_pooled_feat(delta_p, box, search_xyz, search_feat)
            return search_pooled_feat_f, search_box, score, delta_p

    def forward(self, search, template, search_box_input, template_box, search_gt_state=None, train=True):
        """

        :param search: point cloud with shape of [B, N, 3]
        :param template: point cloud with shape of [B, N, 3]
        :param search_box_input: search-region RoI with shape of [B, m, 7]
        :param template_box: tempalte RoI with shape of [B, m, 7]
        :return:
        """

        # extract feature
        template_xyz, template_feat, template_mid_xyz, template_mid_feat = self.backbone_net(template, [256, 128,64])  # [B, N, 3], [B, C, N]
        search_xyz, search_feat, search_mid_xyz, search_mid_feat = self.backbone_net(search, [512, 256, 128])
        # template_xyz, template_feat, template_mid_xyz, template_mid_feat, _ = self.backbone_net(template)
        # search_xyz, search_feat, search_mid_xyz, search_mid_feat, search_seed_inds = self.backbone_net(search)

        # template_xyz_feat = self.canonical_xyz_mlp(template_xyz.transpose(1, 2).contiguous())
        # search_xyz_feat = self.canonical_xyz_mlp(search_xyz.transpose(1, 2).contiguous())
        #
        # template_feat = torch.cat((template_feat, template_xyz_feat), dim=1)
        # search_feat = torch.cat((search_feat, search_xyz_feat), dim=1)

        # transformer
        # template_feat_, search_feat_ = self.transformer(template_feat, search_feat)
        # template_feat = template_feat + template_feat_
        # # search_feat = search_feat + search_feat_   # v1.2
        # search_feat = template_feat + search_feat_ # v1.1

        # classification: target or non-target
        template_semantic = self.semantic_cls(template_feat).transpose(1, 2).contiguous()
        search_semantic = self.semantic_cls(search_feat).transpose(1, 2).contiguous()  # [B, N, 1]

        # proposal: regression
        ## for NMS proposals
        # search_box, _, proposals, proposals_center = self.proposal_layer(search_xyz,
        #                                                                  search_feat,
        #                                                                  search_semantic,
        #                                                                  template_box)  # [B, m, 7]
        # vote_pointwise_xyz = None

        ## for vote proposal
        vote_pointwise_xyz, proposals, proposals_center, search_semantic = self.proposal_layer(search_mid_xyz,
                                                                                               search_mid_feat,
                                                                                               template_mid_xyz,
                                                                                               template_mid_feat)
        # proposals = search_box_input[:,:,[0, 1, 2, 6]] # only for gaussian test
        # proposals_center = search_box_input[:,:,[0, 1, 2]]

        num_prop_vote = proposals.shape[1]
        search_box = template_box.repeat(1, num_prop_vote, 1)
        search_box[:, :, [0, 1, 2, 6]] = proposals

        if not self.learn_R:
            search_box = torch.cat([search_box_input, search_box], dim=1)

        search_box_list = [search_box[:, -num_prop_vote:, :]]
        score_list, delta_p_list = [], []
        nseed_per_frame = search_box.shape[1]

        # crop feature inside box from 'template', shape: [B, out_x, out_y, out_z, C]
        # roiPool(PointRcnn CVPR2019), RoIAwarePool(partA2), RoIGrid(PVRCNN) or pointPool(STD ICCV2019)??
        template_pooled_feat = self.roi_aware_pool(template_xyz, template_box, template_feat)
        search_pooled_feat = self.roi_aware_pool(search_xyz, search_box, search_feat)

        # template_pooled_feat_f = torch.flatten(template_pooled_feat, 1, 3).transpose(1,2)
        # search_pooled_feat_f = torch.flatten(search_pooled_feat, 1,3).transpose(1,2) #[B, C, N]
        # # fusing
        # fused_feat, C_global_feat, T_global_feat = self.xcorr(search_pooled_feat_f, template_pooled_feat_f) # [B, C, 1]

        template_pooled_feat_f = torch.flatten(template_pooled_feat, 1).unsqueeze(-1) # [B*1, C*N, 1]
        search_pooled_feat_f = torch.flatten(search_pooled_feat, 1).unsqueeze(-1)  # [B*m, C*N, 1]

        T_proposal_feat = self.proposal_feat(template_pooled_feat_f)  # [B*1, 256, 1]
        T_similar_feat = self.similar_feat(T_proposal_feat)
        T_reg_feat = self.reg_feat(T_proposal_feat)

        T_similar_feat = T_similar_feat.unsqueeze(1).repeat(1, nseed_per_frame, 1, 1)
        T_similar_feat = T_similar_feat.flatten(0, 1)  # [B*m, 256, 1]

        T_reg_feat = T_reg_feat.unsqueeze(1).repeat(1, nseed_per_frame, 1, 1)
        T_reg_feat = T_reg_feat.flatten(0, 1)  # [B*m, 256, 1]

        for i in range(self.num_R):
            search_pooled_feat_f, search_box, score, delta_p = self.progressive(i, nseed_per_frame,
                                                                                search_pooled_feat_f, search_box,
                                                                                search_xyz, search_feat,
                                                                                T_similar_feat, T_reg_feat,
                                                                                search_gt_state,
                                                                                self.learn_R, train)
            search_box_list.append(search_box[:, -num_prop_vote:, :])  # [[B, m, 7], ...]
            score_list.append(score[:, -num_prop_vote:, :])  # [[B, m, 1], ...]
            delta_p_list.append(delta_p[:, -num_prop_vote:, :])  # [[B, m, 4], ...]

        # calculating similarity for last step
        C_proposal_feat = self.proposal_feat(search_pooled_feat_f)  # [B*m, 256, 1]
        C_similar_feat = self.similar_feat(C_proposal_feat)  # [B*m, 256, 1]

        # calculating similarity
        # score = F.cosine_similarity(T_similar_feat,
        #                             C_similar_feat,
        #                             dim=1)  # [B*m, 1]
        # score = score.view(-1, nseed_per_frame, 1)

        fused_similar_feat = torch.cat((C_similar_feat,T_similar_feat), dim=1)
        # fused_similar_feat = C_similar_feat
        score = self.similar_pred(fused_similar_feat)
        score = score.view(-1, nseed_per_frame, 1)

        score_list.append(score[:, -num_prop_vote:, :])
        search_box_all = torch.cat(search_box_list, dim=1)  # [B, m*(num_R+1), 7]
        score_all = torch.cat(score_list, dim=1)  # [B, m*(num_R+1), 1]
        delta_p_all = torch.cat(delta_p_list, dim=1)  # [B, m*num_R, 4]
        self.first = False

        return search_box_all, delta_p_all, score_all, template_semantic, search_semantic, \
               vote_pointwise_xyz, proposals, proposals_center#, search_seed_inds

    def get_pooled_feat(self, delta_p, prev_box, search_xyz, search_feat):

        # warp box
        new_box = self.warp_box(delta_p, prev_box)

        # crop feature inside box from 'search_feat', shape:[B, out_x, out_y, out_z, C]
        search_pooled_feat = self.roi_aware_pool(search_xyz, new_box, search_feat)
        search_pooled_feat_f = torch.flatten(search_pooled_feat, 1).unsqueeze(-1)  # [B, CxN, 1]
        return search_pooled_feat_f, new_box

    def warp_box(self, delta_p, prev_box):

        new_box = prev_box.clone()
        new_box[:, :, 0] = prev_box[:, :, 0] + delta_p[:, :, 0]
        new_box[:, :, 1] = prev_box[:, :, 1] + delta_p[:, :, 1]
        new_box[:, :, 2] = prev_box[:, :, 2] + delta_p[:, :, 2]
        new_box[:, :, 6] = prev_box[:, :, 6] + delta_p[:, :, -1]
        return new_box


class Proposal_Layer_vote(nn.Module):

    def __init__(self, num_proposal=64, use_xyz=True, objective=False):

        super(Proposal_Layer_vote, self).__init__()

        self.num_proposal = num_proposal

        self.cosine = nn.CosineSimilarity(dim=1)
        self.mlp = pt_utils.SharedMLP([4+256,256,256,256], bn=True)

        self.FC_layer_cla = (
            pt_utils.Seq(256)
                .conv1d(256, bn=True)
                .conv1d(256, bn=True)
                .conv1d(1, activation=None))
        self.fea_layer = (pt_utils.Seq(256)
                          .conv1d(256, bn=True)
                          .conv1d(256, activation=None))

        self.vote_layer = (pt_utils.Seq(3 + 256)
                           .conv1d(256, bn=True)
                           .conv1d(256, bn=True)
                           .conv1d(3 + 256, activation=None))

        self.vote_aggregation = PointnetSAModule(radius=0.3,
                                                 nsample=16,
                                                 mlp=[1 + 256, 256, 256, 256],
                                                 use_xyz=use_xyz)

        self.FC_proposal = (pt_utils.Seq(256)
                            .conv1d(256, bn=True)
                            .conv1d(256, bn=True)
                            .conv1d(3 + 1, activation=None))

    def xcorr(self, x_label, x_object, template_xyz):

        B = x_object.size(0)
        f = x_object.size(1)
        n1 = x_object.size(2)
        n2 = x_label.size(2)
        final_out_cla = self.cosine(x_object.unsqueeze(-1).expand(B,f,n1,n2), x_label.unsqueeze(2).expand(B,f,n1,n2))

        fusion_feature = torch.cat((final_out_cla.unsqueeze(1),
                                    template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B,3,n1,n2)),
                                   dim = 1)

        fusion_feature = torch.cat((fusion_feature,x_object.unsqueeze(-1).expand(B,f,n1,n2)),dim = 1)

        fusion_feature = self.mlp(fusion_feature)

        fusion_feature = F.max_pool2d(fusion_feature, kernel_size=[fusion_feature.size(2), 1])
        fusion_feature = fusion_feature.squeeze(2)
        fusion_feature = self.fea_layer(fusion_feature)

        return fusion_feature

    def forward(self, search_xyz, search_feature, template_xyz, template_feature):

        fusion_feature = self.xcorr(search_feature, template_feature, template_xyz)
        # fusion_feature = search_feature

        estimation_cla = self.FC_layer_cla(fusion_feature).squeeze(1)

        score = estimation_cla.sigmoid()

        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(),fusion_feature),dim = 1)

        # score = estimation_cla.squeeze(-1).sigmoid()
        #
        # fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(), search_feature), dim=1)

        offset = self.vote_layer(fusion_xyz_feature)
        vote = fusion_xyz_feature + offset
        vote_xyz = vote[:, 0:3, :].transpose(1, 2).contiguous()
        vote_feature = vote[:, 3:, :]

        vote_feature = torch.cat((score.unsqueeze(1), vote_feature), dim=1)

        center_xyzs, proposal_features = self.vote_aggregation(vote_xyz, vote_feature, self.num_proposal)

        proposal_offsets = self.FC_proposal(proposal_features)

        shift_center_xyz = proposal_offsets[:, 0:3, :] + center_xyzs.transpose(1, 2).contiguous()

        estimation_boxs = torch.cat((shift_center_xyz, proposal_offsets[:, 3:4, :]), dim=1)

        # proposals = template_box.repeat(1, self.num_proposal, 1)
        # proposals[:, :, [0, 1, 2, 6]] = estimation_boxs.transpose(1, 2).contiguous()

        estimation_boxs = estimation_boxs.transpose(1, 2).contiguous()

        return vote_xyz, estimation_boxs, center_xyzs, estimation_cla.unsqueeze(1)


class Proposal_Layer(nn.Module):
    """
    generate some candidate bounding boxes
    """

    def __init__(self, num_proposal, use_xyz=True):
        super(Proposal_Layer, self).__init__()
        self.num_proposal_pre = 900
        self.num_proposal = num_proposal
        # self.feat_sample = PointnetSAModuleFeat(mlp=[256, 256, 256, 256],
        #                                         radius=0.3,
        #                                         nsample=32,
        #                                         bn=True,
        #                                         use_xyz=use_xyz)
        self.reg = (pt_utils.Seq(256)
                    .conv1d(128, bn=True)
                    .dropout(0.5)
                    .conv1d(4, activation=None))

    def forward(self, search_xyz, search_feature, estimation_cla, template_box):
        """

        :param search_feature: [B, C, N]
        :param estimation_cla: [B, N, 1]
        :param search_xyz: [B, N, 3]
        :param template_box: [B, 1, 7]
        :return:
        """
        batch_size, _, npoints = search_feature.shape

        offset = self.reg(search_feature) #[B, 4, N]
        offset = offset.transpose(1, 2).contiguous()
        vote_xyz = search_xyz + offset[:, :, 0:3]


        proposals = template_box.repeat(1, npoints, 1)
        proposals[:, :, [0, 1, 2]] = vote_xyz
        proposals[:, :, 6] = offset[:, :, 3]

        scores = estimation_cla.sigmoid().squeeze(-1)
        _, sorted_idxs = torch.sort(scores, dim=1, descending=True)

        ret_bbox3d = scores.new(batch_size, self.num_proposal, 7).zero_()
        ret_scores = scores.new(batch_size, self.num_proposal).zero_()
        for k in range(batch_size):
            scores_single = scores[k]
            proposals_single = proposals[k]
            order_single = sorted_idxs[k]

            # scores_single, proposals_single = self.distance_based_proposal(scores_single,
            #                                                                proposals_single,
            #                                                                order_single)
            scores_single, proposals_single = self.score_based_proposal(scores_single,
                                                                        proposals_single,
                                                                        order_single)
            proposals_tot = proposals_single.size(0)
            ret_bbox3d[k, :proposals_tot] = proposals_single
            ret_scores[k, :proposals_tot] = scores_single

        center_xyzs = ret_bbox3d[:, :, [0, 1, 2]]
        estimation_boxs = ret_bbox3d[:, :, 6:]

        return ret_bbox3d, ret_scores, proposals[:,:, [0,1,2,6]], center_xyzs

    def distance_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        nms_range_list = [-10.0, 10.0]
        pre_tot_top_n = self.num_proposal_pre
        pre_top_n_list = [0, int(pre_tot_top_n * 0.7), pre_tot_top_n - int(pre_tot_top_n * 0.7)]

        post_tot_top_n = self.num_proposal
        post_top_n_list = [0, int(post_tot_top_n * 0.7), post_tot_top_n - int(post_tot_top_n * 0.7)]

        scores_single_list, proposals_single_list = [], []

        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]

        dist = proposals_ordered[:, 2]
        first_mask = (dist > nms_range_list[0]) & (dist <= nms_range_list[1])
        for i in range(1, len(nms_range_list)):
            # get proposal distance mask
            dist_mask = ((dist > nms_range_list[i - 1]) & (dist <= nms_range_list[i]))

            if dist_mask.sum() != 0:
                # this area has points
                # reduce by mask
                cur_scores = scores_ordered[dist_mask]
                cur_proposals = proposals_ordered[dist_mask]

                # fetch pre nms top K
                cur_scores = cur_scores[:pre_top_n_list[i]]
                cur_proposals = cur_proposals[:pre_top_n_list[i]]
            else:
                assert i == 2, '%d' % i
                # this area doesn't have any points, so use rois of first area
                cur_scores = scores_ordered[first_mask]
                cur_proposals = proposals_ordered[first_mask]

                # fetch top K of first area
                cur_scores = cur_scores[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
                cur_proposals = cur_proposals[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]

            # oriented nms
            keep_idx, _ = nms_normal_gpu(cur_proposals, cur_scores, 0.85)

            # Fetch post nms top k
            keep_idx = keep_idx[:post_top_n_list[i]]

            scores_single_list.append(cur_scores[keep_idx])
            proposals_single_list.append(cur_proposals[keep_idx])

        scores_single = torch.cat(scores_single_list, dim=0)
        proposals_single = torch.cat(proposals_single_list, dim=0)
        return scores_single, proposals_single

    def score_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]

        # pre nms top K
        cur_scores = scores_ordered[:self.num_proposal_pre]
        cur_proposals = proposals_ordered[:self.num_proposal_pre]

        keep_idx, _ = nms_gpu(cur_proposals, cur_scores, 0.85)

        # Fetch post nms top k
        keep_idx = keep_idx[:self.num_proposal]

        return cur_scores[keep_idx], cur_proposals[keep_idx]
