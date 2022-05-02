import os
import numpy as np
import torch
from torch.nn import functional as F
from src.common import compute_iou
from src.utils import common_util, plushsim_util
from src.training import BaseTrainer
from sklearn.metrics import roc_curve
from scipy import interp
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from src.utils.plushsim_util import find_nn_cpu, find_emd_cpu

class PlushTrainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, cfg, device=None, vis_dir=None, ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.vis_dir = vis_dir
        self.threshold = cfg['test']['threshold']
        self.pos_weight = torch.FloatTensor([cfg['training']['pos_weight']]).to(device)
        if 'corr_dim' in cfg['model']['decoder_kwargs'] and cfg['model']['decoder_kwargs']['corr_dim'] > 0:
            self.contrastive_threshold = cfg['loss']['contrastive_threshold']
            self.use_geodesics = cfg['loss']['use_geodesics']
            self.loss_type = cfg['loss']['type']

            self.contrastive_coeff_neg = cfg['loss'].get('contrastive_coeff_neg', 1.)
            self.contrastive_neg_thres = cfg['loss'].get('contrastive_neg_thres', 1.)
            self.contrastive_coeff_pos = cfg['loss'].get('contrastive_coeff_pos', 1.)
            self.contrastive_pos_thres= cfg['loss'].get('contrastive_pos_thres', 0.1)
            self.scale_with_geodesics = cfg['loss'].get('scale_with_geodesics', False)

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        self.max_thres = 0.2
        self.discretization = 1000
        self.base_fpr = np.linspace(0,1,101)
        self.base_thres = np.linspace(0,self.max_thres,self.discretization)

    def train_step(self, data, it):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        losses = self.compute_loss(data, it)
        loss = 0
        for v in losses.values():
            loss += v
        loss.backward()
        self.optimizer.step()

        return {k:v.item() for k,v in losses.items()}
    
    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)
        agg_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict, agg_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)
            for k, v in agg_step_dict.items():
                agg_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        # - shape completion ROC
        figs = {}
        if 'tpr' in agg_list:
            figs['OCC_ROC'] = self._get_shape_completion_ROC(agg_list['tpr'])
        if 'fmr_hits' in agg_list:
            fmr = np.array(agg_list['fmr_hits'])
            idx01 = int(0.01 * (self.discretization-1) / self.max_thres)
            idx02 = int(0.02 * (self.discretization-1) / self.max_thres)
            idx05 = int(0.05 * (self.discretization-1) / self.max_thres)
            idx10 = int(0.10 * (self.discretization-1) / self.max_thres)
            eval_dict['FMR.01m_5%'] = np.mean(fmr[:,idx01] > 0.05)
            eval_dict['FMR.02m_5%'] = np.mean(fmr[:,idx02] > 0.05)
            eval_dict['FMR.05m_5%'] = np.mean(fmr[:,idx05] > 0.05)
            eval_dict['FMR.10m_5%'] = np.mean(fmr[:,idx10] > 0.05)
            fmr_std = fmr.std(axis=0)
            eval_dict['FMR.01m_5%_std'] = fmr_std[idx01]
            eval_dict['FMR.02m_5%_std'] = fmr_std[idx02]
            eval_dict['FMR.05m_5%_std'] = fmr_std[idx05]
            eval_dict['FMR.10m_5%_std'] = fmr_std[idx10]
            for tau2 in np.linspace(0.01,0.2,5):
                figs[f'FMR_tau1_wrt_tau2={tau2:.3f}']= self._get_FMR_curve_tau1(fmr, tau2=tau2)
            figs['FMR_tau1']= self._get_FMR_curve_tau1(fmr)
            for tau1 in np.linspace(0.01,0.1,5):
                figs[f'FMR_tau2_wrt_tau1={tau1:.3f}']= self._get_FMR_curve_tau2(fmr, tau1=tau1)
        #ax.scatter(fpr, tpr, s=100, alpha=0.5, color="blue")
        if 'pair_dist' in agg_list:
            all_dists = np.concatenate(agg_list['pair_dist'])
            eval_dict['pair_dist'] = all_dists.mean()
            eval_dict['pair_dist_std'] = all_dists.std()
            figs['dist_hist'] = self._get_pair_distance_histogram(all_dists) 
        return eval_dict, figs

    def _get_pair_distance_histogram(self, all_dists):
        fig, ax = plt.subplots(figsize=(10,7))
        counts, bins, patches = ax.hist(all_dists, density=True, bins=40)  # density=False would make counts
        ax.set_ylabel('Density')
        ax.set_xlabel('Pair Distance')
        return fig
    
    def _get_shape_completion_ROC(self, tpr):
        tprs = np.array(tpr)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = np.maximum(mean_tprs - std, 0)
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(self.base_fpr, mean_tprs, 'b')
        ax.fill_between(self.base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
        ax.plot([0, 1], [0, 1],'r--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        return fig

    def _get_FMR_curve_tau2(self, fmrs, tau1=0.1):
        idx05 = int(tau1 * (self.discretization-1) / self.max_thres)
        # fix tau 1
        means = []
        tau1_min = 0.001
        tau1_max = 0.25
        tau1_ticks = np.linspace(tau1_min, tau1_max, 1000)
        for t in tau1_ticks:
            means.append(np.mean(fmrs[:,idx05] > t, axis=0)) 
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(tau1_ticks, means, 'b')
        ax.set_xlim([tau1_min, tau1_max])
        ax.set_ylim([0.0, 1.0])
        ax.set_ylabel('Feature Match Recall')
        ax.set_xlabel('Inlier Ratio threshold')
        return fig

    def _get_FMR_curve_tau1(self, fmrs, tau2=0.05):
        # tau2 = 0.05 is the inlier ratio 
        # fix tau 2
        mean_fmrs = np.mean(fmrs > tau2, axis=0)
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(self.base_thres, mean_fmrs, 'b')
        ax.set_xlim([0.0, self.max_thres])
        ax.set_ylim([0.0, 1.0])
        ax.set_ylabel('Feature Match Recall')
        ax.set_xlabel('Inlier Distance Threshold')
        return fig
    
    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        for k,v in data.items():
            data[k] = v.to(device)

        eval_dict = {} 
        agg = {}
        idx = data['idx'].item()
        # Compute iou
        with torch.no_grad():
            outputs =  self.model(data)

            gt_occ = data['sampled_occ']
            B,_,N = gt_occ.shape
            gt_occ = gt_occ.reshape([B*2, N])
            occ_iou_np = (gt_occ >= 0.5).cpu().numpy()
            occ_iou_hat_np = (outputs['occ'].probs >= self.threshold).cpu().numpy()
            iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
            eval_dict['iou'] = iou
            eval_dict[f'iou_{self.threshold}'] = iou

            occ_iou_hat_np_2 = (outputs['occ'].probs >= 0.5).cpu().numpy()
            iou = compute_iou(occ_iou_np, occ_iou_hat_np_2).mean()
            eval_dict['iou_0.5'] = iou

            intermediate = (self.threshold + 0.5) / 2
            occ_iou_hat_np_3 = (outputs['occ'].probs >= intermediate).cpu().numpy()
            iou = compute_iou(occ_iou_np, occ_iou_hat_np_3).mean()
            eval_dict[f'iou_{intermediate}'] = iou

            if 'flow' in outputs:
                gt_flow = data['sampled_flow']
                gt_flow = gt_flow.reshape([B*2, N, 3])
                constant = torch.from_numpy(np.array((12.,12.,4.)) / 10. / (1.1,1.1,1.1)).float().cuda()
                loss_flow = F.mse_loss(
                    outputs['flow'] * constant, 
                    gt_flow * constant, 
                reduction='none')
                eval_dict['flow_all_field'] = loss_flow.sum(-1).mean().item()
                loss_flow_np = loss_flow.sum(-1).cpu().numpy()
                loss_flow_pos = loss_flow_np[occ_iou_np]
                # if empty scene, no flow of the object will be present 
                if len(loss_flow_pos) > 0:
                    eval_dict['flow'] = loss_flow_pos.mean()


            gt_pts = data['sampled_pts'].reshape([B*2, N, 3]).cpu().numpy()
            if 'flow' in outputs:
                flow_vis_mean = []
                for i in range(B*2):
                    gt_occ_pts = gt_pts[i][occ_iou_np[i]] * (1200, 1200, 400) / (1.1,1.1,1.1) + (0,0,180)
                    vis_idx = plushsim_util.render_points(gt_occ_pts,
                                                        plushsim_util.CAM_EXTR,
                                                        plushsim_util.CAM_INTR,
                                                        return_index=True)
                    vis_pts = gt_occ_pts[vis_idx]
                    flow_vis_mean.append(loss_flow_np[i][occ_iou_np[i]][vis_idx].mean())
                eval_dict['flow_only_vis'] = np.mean(flow_vis_mean)

            if idx % 10000 == 9999:
                # do expensive evaluations
                # occupancy ROC curve
                fpr, tpr, _ = roc_curve(occ_iou_np.flatten(), 
                                        outputs['occ'].probs.cpu().numpy().flatten())
                base_fpr = np.linspace(0, 1, 101)
                tpr = interp(base_fpr, fpr, tpr)
                tpr[0] = 0.0
                agg['tpr'] = tpr
                f1 = []
                for i in range(B*2):
                    gt_occ_pts = common_util.subsample_points(gt_pts[i][occ_iou_np[i]], return_index=False)
                    pred_pts = common_util.subsample_points(gt_pts[i][occ_iou_hat_np[i]], return_index=False)
                    f1.append(common_util.f1_score(pred_pts, gt_occ_pts))
                f1 = np.array(f1)
                f1score, precision, recall = f1.mean(axis=0)
                eval_dict['f1'] = f1score
                eval_dict['precision'] = precision
                eval_dict['recall'] = recall

                if 'corr' in outputs:
                    # data prep corr
                    corr_f = outputs['corr']
                    num_pairs = corr_f.shape[1]
                    gt_match = np.arange(num_pairs)
                    src_f = corr_f[0].cpu().numpy()
                    tgt_f = corr_f[1].cpu().numpy()
                    # data prep pts 
                    pts = data['sampled_pts'].cpu().numpy().squeeze()
                    src_pts = pts[0][:num_pairs] * (12,12,4) / (1.1,1.1,1.1)
                    tgt_pts = pts[1][:num_pairs] * (12,12,4) / (1.1,1.1,1.1)
                    # normalize points to maximum length of 1.
                    tgt_pts = tgt_pts / np.ptp(tgt_pts, axis=0).max()
                    _, nn_inds_st = find_emd_cpu(src_f, tgt_f)
                    # doing Feature-match recall.
                    eval_dict['match_exact'] = np.mean(gt_match == nn_inds_st)

                    dist_st = np.linalg.norm(tgt_pts - tgt_pts[nn_inds_st], axis=1)
                    eval_dict['match_0.05'] = np.mean(dist_st < 0.05)
                    eval_dict['match_0.1'] = np.mean(dist_st < 0.1)
                    hits = np.array([np.mean(dist_st < f) for f in self.base_thres])
                    agg['fmr_hits'] = hits
                    agg['pair_dist'] = dist_st

            return eval_dict, agg

    def compute_loss(self, data, it):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        for k,v in data.items():
            data[k] = v.to(device)
        outputs =  self.model(data)

        loss = {}
        eval_dict = {}
        # Occupancy Loss
        if 'occ' in outputs:
            # gt points
            gt_occ = data['sampled_occ']
            B,_,N = gt_occ.shape
            gt_occ = gt_occ.reshape([B*2, N])
            occ_iou_np = (gt_occ >= 0.5).cpu().numpy()
            # pred
            logits = outputs['occ'].logits
            loss_i = F.binary_cross_entropy_with_logits(
                logits, gt_occ, reduction='none', pos_weight=self.pos_weight)
            loss['occ'] = loss_i.mean()
            # eval infos
            occ_iou_hat_np = (outputs['occ'].probs >= self.threshold).cpu().numpy()
            iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
            eval_dict['iou'] = iou

        if 'flow' in outputs : 
            gt_occ = data['sampled_occ']
            B,_,N = gt_occ.shape
            gt_occ = gt_occ.reshape([B*2, N])
            mask = (gt_occ > 0.5).bool()
            gt_flow = data['sampled_flow']
            gt_flow = gt_flow.reshape([B*2, N, 3])
            flow_gt_0 = gt_flow[~mask]
            flow_gt_1 = gt_flow[mask]
            flow_pred = outputs['flow']
            flow_pred_0 = flow_pred[~mask]
            flow_pred_1 = flow_pred[mask]
            loss['flow'] = F.mse_loss(flow_pred_1, flow_gt_1) + 0.01 * F.mse_loss(flow_pred_0, flow_gt_0)

        if 'corr' in outputs:
            dist_vec = data['geo_dists']
            corr_f = outputs['corr']
            src_f = corr_f[0]
            src_pos = src_f[dist_vec <= self.contrastive_threshold]
            num_positive = (dist_vec <= self.contrastive_threshold).sum()
            
            tgt_f = corr_f[1]
            tgt_pos = tgt_f[dist_vec <= self.contrastive_threshold]
            if self.loss_type == "contrastive":
                if num_positive > 0:
                    src_neg = src_f[dist_vec >  self.contrastive_threshold]
                    tgt_neg = tgt_f[dist_vec >  self.contrastive_threshold]
                    # Positive loss
                    pos_loss = F.relu(((src_pos - tgt_pos).pow(2).sum(1) + 1e-4).sqrt() 
                                    - self.contrastive_pos_thres).pow(2)
                    pos_loss_mean = pos_loss.mean() 
                    loss['contrastive_pos'] = self.contrastive_coeff_pos * pos_loss_mean 

                # Negative loss
                neg_dist = (dist_vec[dist_vec > self.contrastive_threshold] 
                            / self.contrastive_threshold).log() + 1.
                neg_dist = torch.clamp(neg_dist, max=2)
                neg_loss = F.relu(neg_dist -
                                    ((src_neg - tgt_neg).pow(2).sum(1) + 1e-4).sqrt()).pow(2) 
                if self.scale_with_geodesics:
                    neg_loss = neg_loss / neg_dist
                neg_loss_mean = neg_loss.mean() 
                loss['contrastive_neg'] = self.contrastive_coeff_neg * neg_loss_mean
        return loss
