import torch 
import torch.nn as nn 
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

import numpy
import os
import sys
import datetime
import argparse

from utils.utils import enumerateWithEstimate
from utils.logconfig import logging
from .datasets import LunaDataset
from .model import LunaModel


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class LunaTrainingApp:
    def __init__(self, sys_argv=None) -> None:
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self) -> LunaModel:
        model = LunaModel()
        if self.use_cuda:
            log.info(f'Using CUDA on {torch.cuda.device_count} devices')
            # Use parallel GPUs if more than one are found
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model
    
    def initOptimizer(self) -> torch.optim:
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
    
    def initTrainDl(self) -> DataLoader:
        train_set = LunaDataset(
            val_stride=10,
            isValSet_bool=False
        )

        batch_size = self.cli_args.batch_size
        # Split work amongst multple GPUs
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return train_dl
    
    def initValDl(self) -> DataLoader:
        val_set = LunaDataset(
            val_stride=10,
            isValSet_bool=True
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_set,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return val_dl
    
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g,
            label_g[:,1],
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:,1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:,1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        return loss_g.mean()
    
    def trainLoop(self, epoch_ndx, train_dl):
       pass
       




    







    