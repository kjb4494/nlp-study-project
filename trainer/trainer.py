from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Union
from datetime import datetime
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from trainer.criterion import CVAELoss
import os
import tqdm
import torch
import torch.nn as nn
import time


class CVAETrainer:
    # model: nn.Module --> 파이썬의 정적 타입 선언 문법임
    def __init__(self, config, model):
        self.model = model
        self.da_types = model.da_vocab
        self.criterion = CVAELoss(config['loss'])
        self.device = config['device']
        self.train_outputs = config['train_output']
        self.valid_outputs = config['valid_output']
        self.test_outputs = config['test_output']
        self.num_samples = config['num_samples']
        self.is_learning_decay = config['is_learning_decay']
        self.is_valid_true = config['is_valid_train']
        self.is_test_multi_da = config['is_test_multi_da']
        self.save_epoch_step = config['save_epoch_step']
        self.output_dir_path = config['output_dir_path']

        # epoch 설정은 선택사항
        self.epoch = config['epoch'] if 'epoch' in config else 0
        self.log_dir_path = os.path.join(self.output_dir_path, config['log_dirname'])
        self.model_dir_path = os.path.join(self.output_dir_path, config['model_dirname'])
        self.test_dir_path = os.path.join(self.output_dir_path, config['test_dirname'])
        self.log_path = os.path.join(self.log_dir_path, config['log_name'])
        self.model_path = os.path.join(self.model_dir_path, config['model_name'])

    def experiment(self, train_data_loader, valid_data_loader, test_data_loader, epoch_start_point=0):
        output_reports = []
        exp_epoch_start_point = 0
        if epoch_start_point > 0:
            model_path = self.model_path.format(epoch_start_point)
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                device = torch.device(self.device)
                self.model.to(device)
                exp_epoch_start_point = epoch_start_point + 1
            else:
                exp_epoch_start_point = 0

        for epoch in range(exp_epoch_start_point, self.epoch):
            train_output = self.train_one_epoch(train_data_loader, epoch)

    def train_one_epoch(self, data_loader, epoch):
        return self._run_one_epoch(self.train_step, data_loader, 'train', epoch, True)

    def _run_one_epoch(self, step_function, data_loader, mode, epoch, is_train):
        step_outputs = []
        iterator = tqdm.tqdm(data_loader, desc=mode)
        current_step_num = epoch * len(iterator)

        for step_id, model_input in enumerate(iterator):
            current_step_num = current_step_num + 1
            # step_output = step_function(model)