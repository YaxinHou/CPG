# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@IMB_ALGORITHMS.register('long_tail_la')
class Long_tail_la(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(Long_tail_la, self).__init__(args, net_builder, tb_log, logger)
        
        if self.num_classes == 10:
            self.tau = 2.5

        if self.num_classes == 100:
            self.tau = 1.5

        if self.num_classes == 101:
            self.tau = 1.5

        # compute lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        
        self.p_hat_lb = torch.from_numpy((lb_class_dist / lb_class_dist.sum()).astype(np.float32)).cuda(args.gpu)


    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb

        # inference and calculate sup losses
        with self.amp_cm():
            inputs = x_lb_w
            outputs = self.model(inputs)
            logits_x_lb = outputs['logits']
            feats_x_lb = outputs['feat']
            
            feat_dict = {'x_lb': feats_x_lb}

            sup_loss = self.ce_loss(logits_x_lb + self.tau * torch.log(self.p_hat_lb), lb, reduction='mean')

            total_loss = sup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         total_loss=total_loss.item())
        return out_dict, log_dict