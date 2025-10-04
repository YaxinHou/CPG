# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
from torchvision import transforms
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.core.utils import get_data_loader
from semilearn.datasets.augmentation import RandAugment
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset

@IMB_ALGORITHMS.register('cpg')
class CPG(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(CPG, self).__init__(args, net_builder, tb_log, logger)

        #warm_up epoch
        self.warm_up = args.warm_up

        # dataset update step
        if args.dataset == 'cifar10':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'cifar100':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'food101':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'svhn':
            self.update_step = 5
            self.memory_step = 5

        # augment r and dim
        self.sim_num = None
        self.feat_aug_r = None
        self.fd = self.model.num_features

        # uniform class feature center
        self.optim_cfc = None
        self.cfcd = self.model.num_features

        # adaptive labeled (include the pseudo labeled) data and its dataloader
        self.current_x = None
        self.current_y = None
        self.current_idx = None
        self.current_noise_y = None
        self.current_one_hot_y = None
        self.current_one_hot_noise_y = None

        self.select_ulb_idx = None
        self.select_ulb_label = None
        self.select_ulb_pseudo_label = None
        self.select_ulb_pseudo_label_distribution = None

        self.adaptive_lb_dest = None
        self.adaptive_lb_dest_loader = None

        self.dataset = args.dataset
        self.data = self.dataset_dict['data']
        self.targets = self.dataset_dict['targets']
        self.noised_targets = self.dataset_dict['noised_targets']
        self.lb_idx =  self.dataset_dict['lb_idx']
        self.ulb_idx =  self.dataset_dict['ulb_idx']

        self.mean, self.std = {}, {}

        self.mean['cifar10'] = [0.485, 0.456, 0.406]
        self.mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
        self.mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
        self.mean['svhn'] = [0.4380, 0.4440, 0.4730]
        self.mean['food101'] = [0.485, 0.456, 0.406]

        self.std['cifar10'] = [0.229, 0.224, 0.225]
        self.std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
        self.std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
        self.std['svhn'] = [0.1751, 0.1771, 0.1744]
        self.std['food101'] = [0.229, 0.224, 0.225]

        if self.dataset == 'food101':
            self.transform_weak = transforms.Compose([
                                # transforms.Resize(args.img_size),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

            self.transform_strong = transforms.Compose([
                                # transforms.Resize(args.img_size),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.RandomHorizontalFlip(),
                                RandAugment(3, 5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])
        else:
            self.transform_weak = transforms.Compose([
                                transforms.Resize(args.img_size),
                                transforms.RandomCrop(args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

            self.transform_strong = transforms.Compose([
                                transforms.Resize(args.img_size),
                                transforms.RandomCrop(args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                RandAugment(3, 5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

        # compute lb dist
        lb_class_dist = [0 for _ in range(self.num_classes)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)

        self.lb_dist = torch.from_numpy(lb_class_dist.astype(np.float32)).cuda(args.gpu)

        # compute select_ulb and ulb dist
        ulb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_ulb'].targets:
            ulb_class_dist[c] += 1
        ulb_class_dist = np.array(ulb_class_dist)

        self.ulb_dist = torch.from_numpy(ulb_class_dist.astype(np.float32)).cuda(args.gpu)

        self.select_ulb_dist = torch.zeros(self.num_classes).cuda(args.gpu)

        # compute lb_select_ulb and lb_ulb dist
        lb_ulb_class_dist = lb_class_dist + ulb_class_dist

        self.lb_ulb_dist = torch.from_numpy(lb_ulb_class_dist.astype(np.float32)).cuda(args.gpu)

        self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # ~self.warm_up ce loss only and not select unlabeled data
            if self.epoch < self.warm_up:
                self.adaptive_lb_dest_loader = self.loader_dict['train_lb']
                self.lb_select_ulb_dist = self.lb_dist
                self.select_ulb_dist = torch.ones(self.num_classes).cuda(self.args.gpu)

            # self.warm_up select unlabeled data but still use labeled data only to compute loss
            elif self.epoch == self.warm_up:
                self.adaptive_lb_dest_loader = self.loader_dict['train_lb']
                self.lb_select_ulb_dist = self.lb_dist
                self.select_ulb_dist = torch.ones(self.num_classes).cuda(self.args.gpu)

                # only for feature
                self.current_idx = self.lb_idx
                self.current_x = self.data[self.lb_idx]
                self.current_y = self.targets[self.lb_idx]
                self.current_noise_y = self.noised_targets[self.lb_idx]

                self.adaptive_feature_dest = BasicDataset(self.current_idx, self.current_x, self.current_y, self.current_noise_y, self.args.num_classes, False, weak_transform=self.transform_weak, strong_transform=self.transform_strong, onehot=False)
                self.adaptive_feature_loader = get_data_loader(self.args, self.adaptive_feature_dest, self.args.batch_size, data_sampler=None, num_workers=self.args.num_workers, drop_last=False)

                # reset current labeled data
                self.current_x = None
                self.current_y = None
                self.current_idx = None
                self.current_noise_y = None

                # get the class feature center
                feature_mean_center = torch.zeros(self.num_classes, self.cfcd).cuda(self.args.gpu)
                with torch.no_grad():
                    for data in self.adaptive_feature_loader:
                        if self.args.noise_ratio > 0:
                            y = data['y_lb_noised']
                        else:
                            y = data['y_lb']
                        x = data['x_lb_w']

                        if isinstance(x, dict):
                            x = {k: v.cuda(self.gpu) for k, v in x.items()}
                        else:
                            x = x.cuda(self.gpu)
                        y = y.cuda(self.gpu)

                        features = self.model(x.detach())['feat']

                        for feat, label in zip(features, y):
                            feature_mean_center[label] = feature_mean_center[label] + feat

                feature_mean_center = torch.div(feature_mean_center, self.lb_select_ulb_dist.unsqueeze(1))

                self.optim_cfc = feature_mean_center

                # generate the sample augment r
                feat_aug_r = torch.zeros(self.num_classes).cuda(self.gpu)
                sim_num = torch.zeros(self.num_classes).cuda(self.gpu)
                with torch.no_grad():
                    for data in self.adaptive_feature_loader:
                        if self.args.noise_ratio > 0:
                            y = data['y_lb_noised']
                        else:
                            y = data['y_lb']
                        x = data['x_lb_w']

                        if isinstance(x, dict):
                            x = {k: v.cuda(self.gpu) for k, v in x.items()}
                        else:
                            x = x.cuda(self.gpu)
                        y = y.cuda(self.gpu)

                        features = self.model(x.detach())['feat']

                        feature_uniform_center_similarity = (torch.sum(torch.mul(features / torch.norm(features, dim=1, keepdim=True), self.optim_cfc[y] / torch.norm(self.optim_cfc[y], dim=1, keepdim=True)), dim=1) + 1.0) / 2.0
                        sim_num[y] += feature_uniform_center_similarity

                self.sim_num = 1 / torch.div(sim_num, self.lb_select_ulb_dist)
                self.feat_aug_r = self.sim_num / torch.sum(self.sim_num)

            # self.warm_up+1~ use labeled (include the pseudo labeled) data and continue select unlabeled data
            # update the labeled (include the pseudo labeled) dataset and labeled (include the pseudo labeled) data distribution and selected unlabeled data distribution
            else:
                if self.epoch % self.memory_step == 0:
                    self.current_x = None
                    self.current_y = None
                    self.current_idx = None
                    self.current_noise_y = None
                    self.current_one_hot_y = None
                    self.current_one_hot_noise_y = None
                    self.select_ulb_pseudo_label_distribution = None

                    # process selected condident unlabeled data
                    # delete the same idx / same data contribution to gradient once
                    select_ulb_idx_to_label = {}
                    select_ulb_idx_to_pseudo_label = {}

                    for ulb_idx, ulb_pseudo_label, ulb_label in zip(self.select_ulb_idx, self.select_ulb_pseudo_label, self.select_ulb_label):
                        if ulb_idx.item() in select_ulb_idx_to_label:
                            select_ulb_idx_to_label[ulb_idx.item()].append(ulb_label.item())
                        else:
                            select_ulb_idx_to_label[ulb_idx.item()] = [ulb_label.item()]

                        if ulb_idx.item() in select_ulb_idx_to_pseudo_label:
                            select_ulb_idx_to_pseudo_label[ulb_idx.item()].append(ulb_pseudo_label.item())
                        else:
                            select_ulb_idx_to_pseudo_label[ulb_idx.item()] = [ulb_pseudo_label.item()]

                    select_ulb_unique_idx = torch.unique(self.select_ulb_idx)

                    mean_number_of_pseudo_label = []

                    for ulb_unique_idx in select_ulb_unique_idx:
                        mean_number_of_pseudo_label.append(len(select_ulb_idx_to_label[ulb_unique_idx.item()]))

                    select_ulb_unique_label = []
                    select_ulb_unique_pseudo_label = []
                    select_ulb_unique_pseudo_label_distribution = []

                    for ulb_unique_idx in select_ulb_unique_idx:
                        ulb_unique_label = select_ulb_idx_to_label[ulb_unique_idx.item()]
                        ulb_unique_pseudo_label = select_ulb_idx_to_pseudo_label[ulb_unique_idx.item()]

                        ulb_unique_pseudo_label_distribution = torch.zeros(self.num_classes)
                        for item in ulb_unique_pseudo_label:
                            ulb_unique_pseudo_label_distribution[item] += 1.0
                        ulb_unique_pseudo_label_distribution = ulb_unique_pseudo_label_distribution / torch.sum(ulb_unique_pseudo_label_distribution)

                        # process the ground-truth label                                                        
                        select_ulb_unique_label.append(torch.tensor([ulb_unique_label[0]]))

                        # process the pseudo-label
                        if len(ulb_unique_pseudo_label) > 12:
                            most_common_label = Counter(ulb_unique_pseudo_label).most_common(1)[0][0]
                            most_common_number = Counter(ulb_unique_pseudo_label).most_common(1)[0][1]
                            if most_common_number > 0.8 * len(ulb_unique_pseudo_label):
                                select_ulb_unique_pseudo_label.append(torch.tensor([most_common_label]))
                            else:
                                select_ulb_unique_pseudo_label.append(torch.tensor([-1]))
                        else:
                            select_ulb_unique_pseudo_label.append(torch.tensor([-1]))

                        # process the pseudo-label distribution
                        select_ulb_unique_pseudo_label_distribution.append(ulb_unique_pseudo_label_distribution.unsqueeze(0))

                    select_ulb_unique_label = torch.cat(select_ulb_unique_label)
                    select_ulb_unique_pseudo_label = torch.cat(select_ulb_unique_pseudo_label)
                    select_ulb_unique_pseudo_label_distribution = torch.cat(select_ulb_unique_pseudo_label_distribution)

                    self.select_ulb_idx = torch.masked_select(select_ulb_unique_idx.cpu(), select_ulb_unique_pseudo_label != -1)
                    self.select_ulb_label = torch.masked_select(select_ulb_unique_label, select_ulb_unique_pseudo_label != -1)
                    self.select_ulb_pseudo_label = torch.masked_select(select_ulb_unique_pseudo_label, select_ulb_unique_pseudo_label != -1)
                    self.select_ulb_pseudo_label_distribution = select_ulb_unique_pseudo_label_distribution[select_ulb_unique_pseudo_label != -1]

                    # # view the distribution of select unlabeled data (include true and false pseudo labeled unlabeled data)
                    # if not os.path.exists(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch))):
                    #     os.makedirs(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch)))
                    #
                    # class_indices = np.arange(self.num_classes)
                    #
                    # bar_width = 0.3
                    # index = class_indices - bar_width / 2
                    #
                    # record_mask_true = torch.zeros(self.num_classes).cpu()
                    # record_mask_false = torch.zeros(self.num_classes).cpu()
                    #
                    # record_mask_true.index_add_(0, self.select_ulb_pseudo_label[self.select_ulb_pseudo_label==self.select_ulb_label], torch.ones_like(self.select_ulb_pseudo_label[self.select_ulb_pseudo_label==self.select_ulb_label], dtype=record_mask_true.dtype))
                    # record_mask_false.index_add_(0, self.select_ulb_pseudo_label[self.select_ulb_pseudo_label!=self.select_ulb_label], torch.ones_like(self.select_ulb_pseudo_label[self.select_ulb_pseudo_label!=self.select_ulb_label], dtype=record_mask_false.dtype))
                    #
                    # self.print_fn('ulb_dist:\n' + np.array_str(np.array(self.ulb_dist.cpu())))
                    # self.print_fn('record_mask_true:\n' + np.array_str(np.array(record_mask_true)))
                    # self.print_fn('record_mask_false:\n' + np.array_str(np.array(record_mask_false)))
                    # self.print_fn('record_mask:\n' + np.array_str(np.array(record_mask_true + record_mask_false)))
                    #
                    # fig = plt.figure(figsize=(8, 6), dpi=1000)
                    #
                    # ax = fig.add_subplot(111)
                    #
                    # bar0 = ax.bar(index, torch.log(self.ulb_dist).tolist(), width=bar_width, color='#ffffff', edgecolor='black', label='GT')
                    # bar1 = ax.bar(index + bar_width, torch.log(record_mask_true).tolist(), width=bar_width, color='#e37663', edgecolor='black', label='TP')
                    # for i in range(self.num_classes):
                    #     if i == 0:
                    #         bar2 = ax.bar(index[i] + bar_width, (record_mask_false * torch.log(record_mask_false + record_mask_true) / (record_mask_false + record_mask_true)).tolist()[i], width=bar_width, bottom=(record_mask_true * torch.log(record_mask_false + record_mask_true) / (record_mask_false + record_mask_true)).tolist()[i], color='#76a4bc', edgecolor='black', label='FP')
                    #     else:
                    #         bar2 = ax.bar(index[i] + bar_width, (record_mask_false * torch.log(record_mask_false + record_mask_true) / (record_mask_false + record_mask_true)).tolist()[i], width=bar_width, bottom=(record_mask_true * torch.log(record_mask_false + record_mask_true) / (record_mask_false + record_mask_true)).tolist()[i], color='#76a4bc', edgecolor='black')
                    #
                    # ax.set_ylim(0, max(max(np.array(torch.log(self.ulb_dist).cpu())), max(np.array(torch.log(record_mask_true + record_mask_false)))) + 1)
                    #
                    # ax.set_xlabel('Class index', fontsize=18)
                    # ax.set_ylabel('Number of samples', fontsize=18)
                    #
                    # ax.set_xticks(class_indices)
                    # ax.set_xticklabels([f'{i}' for i in class_indices], fontsize=18, rotation=0)
                    #
                    # sample_indices = np.arange(0, max(max(np.array(torch.log(self.ulb_dist).cpu())), max(np.array(torch.log(record_mask_true + record_mask_false)))) + 1, 3)
                    #
                    # ax.set_yticks(sample_indices)
                    # ax.set_yticklabels([f'$e^{{{int(i)}}}$' for i in sample_indices], fontsize=18, rotation=0)
                    #
                    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=2.5, handleheight=0.8, borderpad=0.2, columnspacing=0.8, fontsize=16, framealpha=0.2)
                    #
                    # plt.subplots_adjust(left=0.15, bottom=0.15)
                    #
                    # plt.savefig(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch), 'mask_true_false.pdf'))
                    # plt.clf()
                    # plt.close()

                    self.select_ulb_dist = torch.zeros(self.num_classes).cuda(self.args.gpu)
                    for item in self.select_ulb_pseudo_label:
                        self.select_ulb_dist[int(item)] += 1

                    self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist

                    self.print_fn('select_ulb_dist:\n' + np.array_str(np.array(self.select_ulb_dist.cpu())))
                    self.print_fn('lb_select_ulb_dist:\n' + np.array_str(np.array(self.lb_select_ulb_dist.cpu())))

                    # update the current labeled and pseudo labeled data
                    self.current_idx = np.concatenate((self.lb_idx, self.select_ulb_idx), axis=0)
                    self.current_x = self.data[self.current_idx]
                    self.current_y = np.concatenate((self.targets[self.lb_idx], self.select_ulb_pseudo_label), axis=0)
                    self.current_noise_y = np.concatenate((self.noised_targets[self.lb_idx], self.select_ulb_pseudo_label), axis=0)
                    current_one_hot_y = np.full((len(self.targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                    current_one_hot_y[np.arange(len(self.targets[self.lb_idx])), self.targets[self.lb_idx]] = 1.0 - self.args.smoothing
                    current_one_hot_noise_y = np.full((len(self.noised_targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                    current_one_hot_noise_y[np.arange(len(self.noised_targets[self.lb_idx])), self.noised_targets[self.lb_idx]] = 1.0 - self.args.smoothing
                    self.current_one_hot_y = np.concatenate((current_one_hot_y, self.select_ulb_pseudo_label_distribution), axis=0)
                    self.current_one_hot_noise_y = np.concatenate((current_one_hot_noise_y, self.select_ulb_pseudo_label_distribution), axis=0)

                    # reset select ulb idx and its pseudo label
                    self.select_ulb_idx = None
                    self.select_ulb_label = None
                    self.select_ulb_pseudo_label = None
                    self.select_ulb_pseudo_label_distribution = None

                if (self.epoch - self.warm_up - self.memory_step) % self.update_step == 0:
                    self.print_fn(str(self.epoch) + ': Update the labeled data.')
                    # construct the current lb_select_ulb data and its dataloader
                    self.adaptive_lb_dest = BasicDataset(self.current_idx, self.current_x, self.current_one_hot_y, self.current_one_hot_noise_y, self.args.num_classes, False, weak_transform=self.transform_weak, strong_transform=self.transform_strong, onehot=False)
                    self.adaptive_lb_dest_loader = get_data_loader(self.args, self.adaptive_lb_dest, self.args.batch_size, data_sampler=self.args.train_sampler, num_iters=self.num_train_iter, num_epochs=self.epochs, num_workers=self.args.num_workers, distributed=self.distributed)

                    # only for feature
                    self.adaptive_feature_dest = BasicDataset(self.current_idx, self.current_x, self.current_y, self.current_noise_y, self.args.num_classes, False, weak_transform=self.transform_weak, strong_transform=self.transform_strong, onehot=False)
                    self.adaptive_feature_loader = get_data_loader(self.args, self.adaptive_feature_dest, self.args.batch_size, data_sampler=None, num_workers=self.args.num_workers, drop_last=False)

                    # reset current labeled (include pseudo labeled) data
                    self.current_x = None
                    self.current_y = None
                    self.current_idx = None
                    self.current_noise_y = None
                    self.current_one_hot_y = None
                    self.current_one_hot_noise_y = None

                    # get the class feature center
                    feature_mean_center = torch.zeros(self.num_classes, self.cfcd).cuda(self.args.gpu)
                    with torch.no_grad():
                        for data in self.adaptive_feature_loader:
                            if self.args.noise_ratio > 0:
                                y = data['y_lb_noised']
                            else:
                                y = data['y_lb']
                            x = data['x_lb_w']

                            if isinstance(x, dict):
                                x = {k: v.cuda(self.gpu) for k, v in x.items()}
                            else:
                                x = x.cuda(self.gpu)
                            y = y.cuda(self.gpu)

                            features = self.model(x.detach())['feat']

                            for feat, label in zip(features, y):
                                feature_mean_center[label] = feature_mean_center[label] + feat

                    feature_mean_center = torch.div(feature_mean_center, self.lb_select_ulb_dist.unsqueeze(1))

                    self.optim_cfc = feature_mean_center

                    # generate the sample augment r
                    feat_aug_r = torch.zeros(self.num_classes).cuda(self.gpu)
                    sim_num = torch.zeros(self.num_classes).cuda(self.gpu)
                    with torch.no_grad():
                        for data in self.adaptive_feature_loader:
                            if self.args.noise_ratio > 0:
                                y = data['y_lb_noised']
                            else:
                                y = data['y_lb']
                            x = data['x_lb_w']

                            if isinstance(x, dict):
                                x = {k: v.cuda(self.gpu) for k, v in x.items()}
                            else:
                                x = x.cuda(self.gpu)
                            y = y.cuda(self.gpu)

                            features = self.model(x.detach())['feat']

                            feature_uniform_center_similarity = (torch.sum(torch.mul(features / torch.norm(features, dim=1, keepdim=True), self.optim_cfc[y] / torch.norm(self.optim_cfc[y], dim=1, keepdim=True)), dim=1) + 1.0) / 2.0
                            sim_num[y] += feature_uniform_center_similarity

                    self.sim_num = 1 / torch.div(sim_num, self.lb_select_ulb_dist)
                    self.feat_aug_r = self.sim_num / torch.sum(self.sim_num)

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.adaptive_lb_dest_loader,
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_lb_s, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb_w, logits_x_lb_s = outputs['logits'][:2 * num_lb].chunk(2)
                aux_logits_x_lb_w, aux_logits_x_lb_s = outputs['aux_logits'][:2 * num_lb].chunk(2)                
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][2 * num_lb:].chunk(2)
                aux_logits_x_ulb_w, aux_logits_x_ulb_s = outputs['aux_logits'][2 * num_lb:].chunk(2)                
                feats_x_lb_w, feats_x_lb_s = outputs['feat'][:2 * num_lb].chunk(2)
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][2 * num_lb:].chunk(2)
            else:
                outs_x_lb_w = self.model(x_lb_w)
                logits_x_lb_w = outs_x_lb_w['logits']
                aux_logits_x_lb_w = outs_x_lb_w['aux_logits']
                feats_x_lb_w = outs_x_lb_w['feat']
                outs_x_lb_s = self.model(x_lb_s)
                logits_x_lb_s = outs_x_lb_s['logits']
                aux_logits_x_lb_s = outs_x_lb_s['aux_logits']
                feats_x_lb_s = outs_x_lb_s['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                aux_logits_x_ulb_s = outs_x_ulb_s['aux_logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    aux_logits_x_ulb_w = outs_x_ulb_w['aux_logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb_w': feats_x_lb_w, 'x_lb_s': feats_x_lb_s, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            if self.epoch < self.warm_up:
                # loss for labeled data
                # compute cross entropy loss
                lb_smooth = torch.zeros(num_lb, self.num_classes).cuda(self.args.gpu)
                lb_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                lb_smooth.scatter_(1, lb.unsqueeze(1), 1.0 - self.args.smoothing)
                lb = lb_smooth

                sup_loss = self.ce_loss(logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb, reduction='mean')

                aux_pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=self.compute_prob(aux_logits_x_ulb_w.detach()), use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                aux_pseudo_label_w_smooth = torch.zeros(self.args.uratio * num_lb, self.num_classes).cuda(self.args.gpu)
                aux_pseudo_label_w_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                aux_pseudo_label_w_smooth.scatter_(1, aux_pseudo_label_w.unsqueeze(1), 1.0 - self.args.smoothing)
                aux_loss = self.ce_loss(aux_logits_x_ulb_s, aux_pseudo_label_w_smooth, reduction='mean') + self.ce_loss(aux_logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb, reduction='mean')

                mask = torch.tensor([False]).cuda(self.args.gpu)
            else:
                # loss for labeled data = self.warm_up / for labeled and pseudo labeled data > self.warm_up
                # augment labeled data
                if self.epoch < self.warm_up + self.memory_step:
                    aug_feats_x_lb_times = (torch.ones_like(lb) * 10.0).int()
                    lb_smooth = torch.zeros(num_lb, self.num_classes).cuda(self.args.gpu)
                    lb_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                    lb_smooth.scatter_(1, lb.unsqueeze(1), 1.0 - self.args.smoothing)
                    lb = lb_smooth
                else:
                    aug_feats_x_lb_times = (10.0 * torch.sort(self.lb_select_ulb_dist).values[self.num_classes // 3] / self.lb_select_ulb_dist).int()[torch.argmax(lb, dim=1)]

                aug_lbs_x_lb = lb
                aug_feats_x_lb = feats_x_lb_w

                for feat, label, aug_time in zip(feats_x_lb_w[aug_feats_x_lb_times != 0], lb[aug_feats_x_lb_times != 0], aug_feats_x_lb_times[aug_feats_x_lb_times != 0]):
                    repeated_feat = torch.cat([feat.unsqueeze(0) for _ in range(aug_time)], dim=0)
                    repeated_label = torch.cat([label.unsqueeze(0) for _ in range(aug_time)], dim=0)
                    repeated_feat_aug_r = (torch.cat([self.feat_aug_r[torch.argmax(label).item()].unsqueeze(0) for _ in range(aug_time)], dim=0)).view(-1, 1)
                    repeated_feat_norm = torch.cat([F.normalize(feat.unsqueeze(0)) for _ in range(aug_time)], dim=0)

                    aug_lbs_x_lb = torch.cat([aug_lbs_x_lb, repeated_label], dim=0)
                    aug_feats_x_lb = torch.cat([aug_feats_x_lb, repeated_feat + torch.randn(aug_time, self.fd).cuda(self.gpu) * repeated_feat_aug_r * repeated_feat_norm], dim=0)

                aug_logits_x_lb = self.model.classifier(aug_feats_x_lb)

                # compute cross entropy loss for labeled data
                sup_loss = self.ce_loss(aug_logits_x_lb + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), aug_lbs_x_lb, reduction='mean')

                # compute probability
                before_refined_select_ulb_dist = torch.where(self.select_ulb_dist <= min(self.lb_dist), 0, self.select_ulb_dist)

                sorted_select_ulb_dist, _ = torch.sort(torch.unique(before_refined_select_ulb_dist))

                if len(sorted_select_ulb_dist) == 1:
                    refined_select_ulb_dist = torch.ones_like(before_refined_select_ulb_dist)
                else:
                    refined_select_ulb_dist = torch.where(before_refined_select_ulb_dist == 0, sorted_select_ulb_dist[1], before_refined_select_ulb_dist)

                probs_x_ulb_w = self.compute_prob((logits_x_ulb_w + torch.log(refined_select_ulb_dist / torch.sum(refined_select_ulb_dist))).detach())
                probs_x_ulb_s = self.compute_prob((logits_x_ulb_s + torch.log(refined_select_ulb_dist / torch.sum(refined_select_ulb_dist))).detach())

                aux_pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=self.compute_prob(aux_logits_x_ulb_w.detach()), use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                aux_loss = self.ce_loss(aux_logits_x_ulb_s, aux_pseudo_label_w, reduction='mean') + self.ce_loss(aux_logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb.argmax(dim=1), reduction='mean')

                # generate unlabeled targets using pseudo label hook
                pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_w, use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                pseudo_label_s = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_s, use_hard_label=self.use_hard_label, T=self.T, softmax=False)

                mask_w_s = pseudo_label_w == pseudo_label_s

                # calculate mask
                mask_w = probs_x_ulb_w.amax(dim=-1).ge(self.p_cutoff * (1.0 - self.args.smoothing))
                mask_s = probs_x_ulb_s.amax(dim=-1).ge(self.p_cutoff * (1.0 - self.args.smoothing))

                mask = torch.logical_and(torch.logical_and(mask_w, mask_s), mask_w_s)

                # update select_ulb_idx and its pseudo_label
                if self.select_ulb_idx is not None and self.select_ulb_pseudo_label is not None and self.select_ulb_label is not None:
                    self.select_ulb_idx = torch.cat([self.select_ulb_idx, idx_ulb[mask]], dim=0)
                    self.select_ulb_label = torch.cat([self.select_ulb_label, y_ulb[mask]], dim=0)
                    self.select_ulb_pseudo_label = torch.cat([self.select_ulb_pseudo_label, pseudo_label_w[mask]], dim=0)
                else:
                    self.select_ulb_idx = idx_ulb[mask]
                    self.select_ulb_label = y_ulb[mask]
                    self.select_ulb_pseudo_label = pseudo_label_w[mask]

            total_loss = sup_loss + aux_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         aux_loss=aux_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--warm_up', int, 30),
            SSL_Argument('--alpha', float, 1.0),
            SSL_Argument('--smoothing', float, 0.1),
        ]