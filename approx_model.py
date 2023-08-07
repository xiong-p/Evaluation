"""model to approximate the single second stage cost"""
import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils import tensorboard
from torch.utils.data import DataLoader

import utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 50
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 10

NUM_LAYERS = 4
NUM_HIDDEN_LIST = [8, 16, 32, 8]
assert len(NUM_HIDDEN_LIST) == NUM_LAYERS


# class ApproxNetwork(nn.Module):
#     def __init__(self, input_dim, num_hidden_list):
#         super(ApproxNetwork, self).__init__()
#         self.num_hidden_list = num_hidden_list
#         prev_hidden = input_dim
#         dic = []
#         for i, hidden in enumerate(self.num_hidden_list):
#             dic.append((f'linear{i}', nn.Linear(prev_hidden, hidden, bias=True)))
#             dic.append((f'activation{i}', nn.GELU()))
#             prev_hidden = hidden
#
#         dic.append((f'linear{len(self.num_hidden_list)}', nn.Linear(prev_hidden, 1, bias=True)))
#         self.net = nn.Sequential(OrderedDict(dic))
#         self.to(DEVICE)
#
#     def forward(self, x):
#         return self.net(x)
class ApproxNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_list):
        super(ApproxNetwork, self).__init__()
        self.num_hidden_list = num_hidden_list
        prev_hidden = input_dim
        dic = []
        for i, hidden in enumerate(self.num_hidden_list):
            dic.append((f'linear{i}', nn.Linear(prev_hidden, hidden, bias=True)))
            dic.append((f'activation{i}', nn.GELU()))
            prev_hidden = hidden

        dic.append((f'linear{len(self.num_hidden_list)}', nn.Linear(prev_hidden, 1, bias=True)))
        dic.append((f'activation{len(self.num_hidden_list)}', nn.Softplus()))
        self.net = nn.Sequential(OrderedDict(dic))
        self.to(DEVICE)

    def forward(self, x):
        return self.net(x)


class ApproxNet:
    def __init__(self, learning_rate, log_dir, input_dim, num_hidden_list):
        self._network = ApproxNetwork(input_dim, num_hidden_list)
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _step(self, X, y):
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        return nn.HuberLoss()(self._network(X), y)

    def train(self, dataloader_train, dataloader_val, writer, num_epochs=1000):
        print(f'Starting training at epoch {self._start_train_step}.')
        for epoch in range(self._start_train_step, num_epochs):
            losses = []
            for X, y in dataloader_train:
                self._optimizer.zero_grad()
                loss = self._step(X, y)
                losses.append(loss.item())
                loss.backward()
                self._optimizer.step()

            if epoch % PRINT_INTERVAL == 0:
                print(
                    f'Epoch {epoch}: '
                    f'loss: {np.mean(losses):.3f} '
                )
                writer.add_scalar('loss/train', np.mean(losses), epoch)

            if epoch % VAL_INTERVAL == 0 and epoch > 0:
                with torch.no_grad():
                    losses_val = []
                    for X, y in dataloader_val:
                        losses_val.append(self._step(X, y).item())
                    loss_val = np.mean(losses_val)
                print(
                    f'Validation: '
                    f'loss: {loss_val:.3f} '
                )
                writer.add_scalar('loss/val', loss_val, epoch)

            if epoch % SAVE_INTERVAL == 0:
                self._save(epoch)

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        losses = []
        for X, y in dataloader_test:
            losses.append(self._step(X, y).item())
        mean = np.mean(losses)
        std = np.std(losses)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(len(dataloader_test))
        print(
            f'Loss over test dataset: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    # def load(self, checkpoint_step):
    #     """Loads a checkpoint.
    #
    #     Args:
    #         checkpoint_step (int): iteration of checkpoint to load
    #
    #     Raises:
    #         ValueError: if checkpoint for checkpoint_step is not found
    #     """
    #     target_path = (
    #         f'{os.path.join(self._log_dir, "state")}'
    #         f'{checkpoint_step}.pt'
    #     )
    #     if os.path.isfile(target_path):
    #         if DEVICE == "cuda":
    #             state = torch.load(target_path)
    #         else:
    #             state = torch.load(target_path, map_location=torch.device('cpu'))
    #         self._network.load_state_dict(state['network_state_dict'])
    #         self._optimizer.load_state_dict(state['optimizer_state_dict'])
    #         self._start_train_step = checkpoint_step + 1
    #         print(f'Loaded checkpoint iteration {checkpoint_step}.')
    #     else:
    #         raise ValueError(
    #             f'No checkpoint for iteration {checkpoint_step} found.'
    #         )
    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step: iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state_")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            if DEVICE == "cuda":
                state = torch.load(target_path)
            else:
                state = torch.load(target_path, map_location=torch.device('cpu'))
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            if isinstance(checkpoint_step, int):
                self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    log_dir = args.log_dir
    epochs = args.epoch

    if log_dir is None:
        log_dir = f'./logs/gen.lr:{learning_rate}.batch_size:{batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    print("using device: ", DEVICE)

    approx_net = ApproxNet(learning_rate, log_dir, 3, NUM_HIDDEN_LIST)

    if args.checkpoint_step > -1:
        approx_net.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    if not args.test:
        dataloader_train = DataLoader(utils.GenDataset('gen_train.csv'), batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(utils.GenDataset('gen_val.csv'), batch_size=batch_size, shuffle=True)
        approx_net.train(dataloader_train, dataloader_val, writer, num_epochs=epochs)
    else:
        dataloader_test = DataLoader(utils.GenDataset('gen_test.csv'), batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a ApproxNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--epoch', type=int, default=5000,
                        help='number of epochs to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    main_args = parser.parse_args()
    main(main_args)