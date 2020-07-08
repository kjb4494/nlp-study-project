from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Union
from datetime import datetime
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import os
import tqdm
import torch
import torch.nn as nn
import time


class Trainer(ABC):
    # model: nn.Module --> 파이썬의 정적 타입 선언 문법임
    def __init__(self, config, model: nn.Module):
        self.config = config
        self.model = model
        self.da_types = model.da_vocab
        self.criterion: Union[nn.Module, Dict[str, nn.Module]] = self._set_criterion(self.config['loss'])

        self.train_outputs = config['train_output']
        self.valid_outputs = config['valid_output']
        self.test_outputs = config['test_output']
        self.num_samples = config['num_samples']
        self.is_learning_decay = config['is_learning_decay']
        self.is_valid_true = config['is_valid_train']
        self.is_test_multi_da = config['is_test_multi_da']
        self.save_epoch_step = config['save_epoch_step']
        self.output_dir_path = config['output_dir_path']
        self.log_dir_path = os.path.join(self.output_dir_path, config['log_dirname'])
        self.model_dir_path = os.path.join(self.output_dir_path, config['model_dirname'])
        self.test_dir_path = os.path.join(self.output_dir_path, config['test_dirname'])
        self.log_path = os.path.join(self.log_dir_path, config['log_name'])
        self.model_path = os.path.join(self.model_dir_path, config['model_name'])

    @abstractmethod
    def _set_criterion(self, loss_config) -> Union[nn.Module, Dict[str, nn.Module]]:
        raise NotImplementedError
