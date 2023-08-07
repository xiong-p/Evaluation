"""model to approximate the single second stage cost"""
import argparse
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.prune as prune

torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 10
PRUNE_STEP = 1000


class ApproxDataset(Dataset):
    def __init__(self, file_path='/Users/xiongjinxin/A-xjx/SRIBD/Evaluation/stage2_pen_active.csv'):
        # convert into PyTorch tensors and remember them
        df = pd.read_csv(file_path)  # , index_col=0)
        self.X = df[df.columns[:-1]].values
        self.y = df[['stage2_pen']].values / 1e5
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx]
        target = self.y[idx]
        return features, target


class ApproxNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_list, l1_lambda=0.0):
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
        self.l1_lambda = l1_lambda

    def forward(self, x):
        return self.net(x)

    def l1_regularization(self):
        l1_loss = 0.0
        for name, param in self.named_parameters():
            if "linear0" in name:
                l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss


class ApproxNet:
    def __init__(self, learning_rate, log_dir, input_dim, num_hidden_list, l1_lambda=0.0):
        self._network = ApproxNetwork(input_dim, num_hidden_list, l1_lambda)
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0
        self._criterion = nn.HuberLoss()
        self.l1_lambda = l1_lambda

    def _step(self, X, y):
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        return self._criterion(self._network(X), y)

    def train(self, dataloader_train, dataloader_val, writer, num_epochs):
        print(f'Starting training at epoch {self._start_train_step}.')
        best_val = float('inf')
        for epoch in range(self._start_train_step, self._start_train_step + num_epochs):
            losses = []
            for X, y in dataloader_train:
                self._optimizer.zero_grad()
                loss = self._step(X, y)
                if self.l1_lambda > 0.0:
                    loss += self._network.l1_regularization()
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
                if loss_val < best_val:
                    best_val = loss_val
                    self._save("best")
                    print(f'New best validation loss: {best_val:.3f}')

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
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    # def _save(self, checkpoint_step, best=False):
    #     """Saves network and optimizer state_dicts as a checkpoint.
    #
    #     Args:
    #         checkpoint_step (int): iteration to label checkpoint with
    #     """
    #     if best:
    #         torch.save(
    #             dict(network_state_dict=self._network.state_dict(),
    #                  optimizer_state_dict=self._optimizer.state_dict()),
    #             f'{os.path.join(self._log_dir, "state")}_0.pt'
    #         )
    #         print('Saved best checkpoint.')
    #         return
    #     torch.save(
    #         dict(network_state_dict=self._network.state_dict(),
    #              optimizer_state_dict=self._optimizer.state_dict()),
    #         f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
    #     )
    #     print('Saved checkpoint.')
    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.
        checkpoint_step can be replaced as any string
        - if int, then save as state123.pt
        - if string, then save as state{string}.pt

        Args:
            checkpoint_step: iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state_")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')

    def prune(self, prune_rate=0.3):
        for name, module in self._network.named_modules():
            # prune 20% of connections in all 2D-conv layers
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.4)
                prune.remove(module, name="weight")

        self._save("pruned")

    # def prune(self, dataloader_train, dataloader_val, finetune_step, prune_rate, writer):
    #     prune.ln_structured(self._network.net.linear0, name="weight", amount=prune_rate, n=1, dim=1)
    #     prune.remove(self._network.net.linear0, name="weight")
    #     self._save(1111)
    #     loss_val_best = float('inf')
    #     for fs in range(finetune_step):
    #         losses = []
    #         for X, y in dataloader_train:
    #             self._optimizer.zero_grad()
    #             loss = self._step(X, y)
    #             losses.append(loss.item())
    #             loss.backward()
    #             self._optimizer.step()
    #
    #         with torch.no_grad():
    #             losses_val = []
    #             for X, y in dataloader_val:
    #                 losses_val.append(self._step(X, y).item())
    #             loss_val = np.mean(losses_val)
    #             if loss_val < loss_val_best:
    #                 loss_val_best = loss_val
    #                 self._save(1222)
    #         prune.remove(self._network.net.linear0, name="weight")
    #         if fs%10 == 0:
    #             print(f'fine-tune {fs} val loss: ', loss_val)
    #             print(f'fine-tune {fs} loss: ', np.mean(losses))
    #         writer.add_scalar('fine_tune/train', np.mean(losses), fs)
    #         writer.add_scalar('fine_tune/val', loss_val, fs)


def main(args):
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    log_dir = args.log_dir
    epochs = args.epoch
    num_hidden_list = args.hidden
    prune_p = args.prune
    l1_lambda = args.l1_lambda

    if log_dir is None:
        log_dir = './logs/'
    else:
        if log_dir[-1] != "/":
            log_dir += "/"
    log_dir = log_dir + f'gen.lr:{learning_rate}.batch_size:{batch_size}.prune:{prune_p}.l1_lambda:{l1_lambda}.hidden:' + "_".join(
            [str(i) for i in num_hidden_list])
    print(f'log_dir: {log_dir}')
    print("using device: ", DEVICE)

    input_dim = 1502
    approx_net = ApproxNet(learning_rate, log_dir, input_dim, num_hidden_list, l1_lambda)

    if args.checkpoint_step > -1:
        approx_net.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    train_file = "../data/stage2_pen_train.csv"
    val_file = "../data/stage2_pen_val.csv"
    test_file = "../data/stage2_pen_test.csv"

    if not args.test:
        dataloader_train = DataLoader(ApproxDataset(train_file), batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(ApproxDataset(val_file), batch_size=batch_size, shuffle=True)
        print(f'Training dataset: {len(dataloader_train)}')
        print(f'Validation dataset: {len(dataloader_val)}')
        approx_net.train(dataloader_train, dataloader_val, writer, num_epochs=epochs)
    else:
        dataloader_test = DataLoader(ApproxDataset(test_file), batch_size=batch_size, shuffle=True)
        approx_net.test(dataloader_test)


def parse_list(input_str):
    try:
        return [int(item) for item in input_str.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid list format: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a ApproxNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='number of epochs to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--hidden', type=parse_list, default=[100, 100],
                        help='number of hidden units per layer as a list')
    parser.add_argument('--prune', type=float, default=0.0,
                        help='the percentage of weights to prune')
    parser.add_argument('--l1_lambda', type=float, default=1.0,
                        help='the hyper-parameter of the regularization for l1 penalty')

    main_args = parser.parse_args()
    main(main_args)
