from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Union
from datetime import datetime
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
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
        self.learning_rate = config['learning_rate']
        self.learning_decay_rate = config['learning_decay_rate']
        self.learning_decay_step = config['learning_decay_step']
        # epoch 설정은 선택사항, 기본값 0
        self.epoch = config['epoch'] if 'epoch' in config else 0

        # criterion -> CVAELoss Forward 함수에 매개변수 전달 가능
        self.criterion = CVAELoss(config['loss'])
        self.optimizer = Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=self.learning_decay_step, gamma=self.learning_decay_rate)

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
            train_output = self._run_one_epoch(
                step_function=self.train_step,
                data_loader=train_data_loader,
                mode='train',
                epoch=epoch,
                is_train=True
            )
            output_report = {"train": train_output}

            if valid_data_loader:
                valid_output = self._run_one_epoch(
                    step_function=self.valid_step,
                    data_loader=valid_data_loader,
                    mode='valid',
                    epoch=epoch,
                    is_train=False
                )

            if test_data_loader:
                test_output = self._run_one_epoch(
                    step_function=self.test_step,
                    data_loader=test_data_loader,
                    mode='test',
                    epoch=epoch,
                    is_train=False
                )

        return output_reports

    def _run_one_epoch(self, step_function, data_loader, mode, epoch, is_train):
        step_outputs = []
        iterator = tqdm.tqdm(data_loader, desc=mode)
        current_step_num = epoch * len(iterator)

        for step_id, model_input in enumerate(iterator):
            current_step_num = current_step_num + 1
            step_output = step_function(model_input, current_step_num)
            # 각 스탭의 리포트 구현은 나중에...
            # self.report_per_step(step_output, step_id, mode, epoch, is_train=is_train)
            step_outputs.append(step_output)

        if is_train and self.is_learning_decay:
            self.scheduler.step()

        return step_outputs

    def train_step(self, model_input, current_step):
        self.optimizer.zero_grad()
        model_input['is_train'] = True
        model_input['num_samples'] = self.num_samples
        model_output = self.model.forward(model_input)

        targets = self.train_outputs

        recorded_model_output = {target: model_output[target] for target in targets}
        loss = self.calculate_loss(
            model_output=model_output,
            model_input=model_input,
            current_step=current_step,
            is_train=True,
            is_valid=False
        )
        self.update_gradient(loss=loss)
        return {
            'model_output': recorded_model_output,
            'loss': loss
        }

    def valid_step(self, model_input, current_step):
        model_input['is_train'] = self.is_valid_true
        model_input['is_train_multiple'] = True
        model_input['is_test_multi_da'] = self.is_test_multi_da
        model_input['num_samples'] = self.num_samples

        targets = self.valid_outputs
        with torch.no_grad():
            model_output = self.model.forward(model_input)
            loss = self.calculate_loss(
                model_output=model_output,
                model_input=model_input,
                current_step=current_step,
                is_train=False,
                is_valid=True
            )
        recorded_model_output = {target: model_output[target] for target in targets}
        return {
            'model_output': recorded_model_output,
            'loss': loss
        }

    def test_step(self, model_input, current_step):
        model_input['is_train'] = False
        model_input['is_test_multi_da'] = self.is_test_multi_da
        model_input['num_samples'] = self.num_samples

        targets = self.test_outputs

        with torch.no_grad():
            model_output = self.model.forward(model_input)
            loss = self.calculate_loss(
                model_output=model_output,
                model_input=model_input,
                current_step=current_step,
                is_train=False,
                is_valid=False
            )
        recored_model_output = {target: model_output[target] for target in targets}
        return {
            'model_input': model_input,
            'model_output': recored_model_output,
            'loss': loss
        }

    def calculate_loss(self, model_output, model_input, current_step, is_train, is_valid):
        return self.criterion(model_output, model_input, current_step, is_train, is_valid)

    def update_gradient(self, loss):
        loss.backward()
        self.optimizer.step()
