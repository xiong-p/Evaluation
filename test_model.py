"""model to approximate the single second stage cost"""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard

NUM_LAYERS = 4
NUM_HIDDEN_LIST = [64, 64, 64, 64]
assert len(NUM_HIDDEN_LIST) == NUM_LAYERS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' #'mps' if torch.backends.mps.is_available() else 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
MAX_EPOCHS = 100
LR = 1e-3

# NUM_TEST_TASKS = 600

class Model:
    def __init__(self, input_size, output_size, log_dir, num_hidden_list, max_epochs=MAX_EPOCHS, lr=LR):
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_list = num_hidden_list

        prev_size = input_size
        parameters = {}
        for i, num_hidden in enumerate(num_hidden_list):
            parameters[f'w{i}'] = nn.init.xavier_uniform_(
                torch.empty(
                    prev_size,
                    num_hidden,
                    requires_grad=True,
                    device=DEVICE
                )
            )

            parameters[f'b{i}'] = nn.init.zeros_(
                torch.empty(
                    num_hidden,
                    requires_grad=True,
                    device=DEVICE
                )
            )
            prev_size = num_hidden

        self.parameters = parameters
        self.max_epochs = max_epochs
        self.lr = lr

        self.optimizer = torch.optim.Adam(
            list(self.parameters.values()),
            lr=self.lr
        )

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.start_train_step = 0

    def forward(self, x, parameters):
        for i in range(len(self.num_hidden_list)):
            x = F.linear(
                input=x,
                weight=parameters[f'w{i}'],
                bias=parameters[f'b{i}']
            )
            x = F.gelu(x)
        return x

    def step(self, x, y, parameters):
        y_pred = self.forward(x, parameters)
        loss = F.mse_loss(y_pred, y)
        return loss, y_pred

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self.log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            if DEVICE == 'cuda':
                state = torch.load(target_path)
            else:
                state = torch.load(target_path, map_location=torch.device('cpu'))
            self.parameters = state['parameters']
            self.lr = state['lr']
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.start_train_step = checkpoint_step + 1
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
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save(
            dict(parameters=self.parameters,
                 lr=self.lr,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self.log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')

