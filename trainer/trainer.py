
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from trainer.criterion import CVAELoss
import os
import tqdm
import torch
import time


class CVAETrainer:
    # model: nn.Module --> 파이썬의 정적 타입 선언 문법임
    def __init__(self, config, model):
        self.model = model
        self.da_types = model.info.da_vocab

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
            start_time = time.time()
            train_output = self._run_one_epoch(
                step_function=self.train_step,
                data_loader=train_data_loader,
                mode='train',
                epoch=epoch,
                is_train=True
            )
            elapsed_time = time.time() - start_time
            self.report_per_epoch(
                metrics=train_output,
                mode_name='Train',
                epoch=epoch,
                elapsed_time=elapsed_time,
                is_train=True
            )
            output_report = {"train": train_output}

            if valid_data_loader:
                start_time = time.time()
                valid_output = self._run_one_epoch(
                    step_function=self.valid_step,
                    data_loader=valid_data_loader,
                    mode='valid',
                    epoch=epoch,
                    is_train=False
                )
                elapsed_time = time.time() - start_time
                self.report_per_epoch(
                    metrics=valid_output,
                    mode_name='Valid',
                    epoch=epoch,
                    elapsed_time=elapsed_time,
                    is_train=False
                )
                output_report['valid'] = valid_output

            if test_data_loader:
                start_time = time.time()
                test_output = self._run_one_epoch(
                    step_function=self.test_step,
                    data_loader=test_data_loader,
                    mode='test',
                    epoch=epoch,
                    is_train=False
                )
                elapsed_time = time.time() - start_time
                self.report_per_epoch(
                    metrics=test_output,
                    mode_name='Test',
                    epoch=epoch,
                    elapsed_time=elapsed_time,
                    is_train=False
                )
                output_report['test'] = test_output

            output_reports.append(output_report)
            if epoch % self.save_epoch_step == 0:
                torch.save(self.model.state_dict(), self.model_path.format(epoch))

        return output_reports

    def _run_one_epoch(self, step_function, data_loader, mode, epoch, is_train):
        step_outputs = []
        iterator = tqdm.tqdm(data_loader, desc=mode)
        current_step_num = epoch * len(iterator)

        for step_id, model_input in enumerate(iterator):
            current_step_num = current_step_num + 1
            step_output = step_function(model_input, current_step_num)
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

    def report_per_epoch(self, metrics, mode_name, epoch=None, elapsed_time=0.0, is_train=True):
        def print_with_logging(file_writer, log_message):
            file_writer.write(log_message + '\n')
            print(log_message)
        log_file_name = self.log_path.format(epoch)
        log_writer = open(log_file_name, 'w')

        # 확인 필요한 코드
        losses = [metric['loss'] for metric in metrics]
        loss = {}
        for key in losses[0].keys():
            loss[key] = 0.0
            for l in losses:
                if type(l[key]) != float:
                    loss[key] += l[key].item()
                else:
                    loss[key] += l[key]
            loss[key] /= len(losses)
        metric = []
        for key, value in loss.items():
            metric.append('{0}: {1:.3f}'. format(key, value))
        metric_str = ', '.join(metric)

        elapsed_time_str = mode_name + ' elapsed Time for Epoch {0} %H:%M:%S'.format(epoch)
        elapsed_time_str = str(time.strftime(elapsed_time_str, time.gmtime(elapsed_time)))
        print_with_logging(log_writer, elapsed_time_str)

        metric_title = 'Metric in {0} set for Epoch #{1}: {2}'.format(mode_name, epoch, metric_str)
        print_with_logging(log_writer, metric_title)

        epoch_ex_str = "========== {0} Examples for Epoch #{1} ==========".format(mode_name, epoch)
        print_with_logging(log_writer, epoch_ex_str)

        for i in range(self.num_samples):
            context_sents = metrics[0]["model_output"]["context_sents"][i]
            for turn_id, turn in enumerate(context_sents):
                context_turn_str = "Context Turn #{0}: {1}".format(turn_id, turn)
                print_with_logging(log_writer, context_turn_str)

            generated_sent = metrics[0]["model_output"]["output_sents"][i]
            generated_sent_str = "Generated (Sample #1): {0}".format(generated_sent)
            print_with_logging(log_writer, generated_sent_str)

            if not is_train:
                for j in range(self.num_samples - 1):
                    sampled_sent = metrics[0]["model_output"]["sampled_output_sents"][j][i]
                    sampled_sent_str = "Sample #{0}: {1}".format(j + 2, sampled_sent)
                    print_with_logging(log_writer, sampled_sent_str)
                if self.is_test_multi_da:
                    for da in self.da_types:
                        multi_da_result = metrics[0]["model_output"]["ctrl_output_sents"][da][i]
                        multi_da_str = "{0} : {1}".format(da, multi_da_result)
                        print_with_logging(log_writer, multi_da_str)

            real_str = metrics[0]["model_output"]["real_output_sents"][i]
            predicted_da = metrics[0]["model_output"]["output_das"][i]
            real_da = metrics[0]["model_output"]["real_output_das"][i]

            print_with_logging(log_writer, "Real Response: {0}".format(real_str))
            print_with_logging(log_writer, "Predicted DA: {0}".format(predicted_da))
            print_with_logging(log_writer, "Real DA: {0}".format(real_da))
        log_writer.close()
