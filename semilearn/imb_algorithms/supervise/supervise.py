# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@IMB_ALGORITHMS.register('supervise')
class Supervise(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(Supervise, self).__init__(args, net_builder, tb_log, logger)

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

            sup_loss = self.ce_loss(logits_x_lb, lb, reduction='mean')

            total_loss = sup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         total_loss=total_loss.item())
        return out_dict, log_dict