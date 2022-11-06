from networks.network import LLL_Net
import importlib
from data_loader import get_loaders
from loggers.exp_logger import MultiLogger
from loggers.logger import Logger
from settings import *
import os
import torch
import random
import numpy as np

cudnn_deterministic = True


def get_method_kwargs(method_paramethers, method_name):
    base_kwargs = dict(nepochs=method_paramethers['NUM_EPOCHS'], eval_on_train=method_paramethers['EVAL_ON_TRAIN'])
    approach_kwargs = dict(approach=method_name,
                           num_exemplars_per_class=method_paramethers['NUM_EXEMPLARS_PER_CLASS'],
                           exem_selection_method=method_paramethers['EXEMPLAR_SELECTION_METHOD'],
                           gridsearch_tasks=method_paramethers['GRIDSEARCH_TASKS'])
    appr_exemplars_dataset_args = dict(num_exemplars=method_paramethers['MAX_NUM_EXEMPLARS'],
                                       exemplar_selection=method_paramethers['EXEMPLAR_SELECTION_METHOD'],
                                       num_exemplars_per_class=method_paramethers['NUM_EXEMPLARS_PER_CLASS'])
    return base_kwargs, approach_kwargs, appr_exemplars_dataset_args


def get_model(method_paramethers):
    # network = getattr(importlib.import_module(name='torchvision.models'), method_paramethers['NETWORK'])
    network = getattr(importlib.import_module(name='networks'), method_paramethers['NETWORK'])
    init_model = network(pretrained=False)
    # set_tvmodel_head_var(init_model)
    net = LLL_Net(init_model, remove_existing_head=False)
    return net


def get_data_loaders(method_paramethers):
    tst_loader = []
    trn_loader, val_loader, tst = get_loaders("args.datasets", None,
                                              method_paramethers['BATCH_SIZE'],
                                              num_workers=method_paramethers['NUM_WORKERS'],
                                              pin_memory=method_paramethers['PIN_MEMORY'])
    tst_loader.append(tst)
    return trn_loader, val_loader, tst_loader


def get_incremental_learning_class(method_name, trn_loader, base_kwargs, logger, appr_exemplars_dataset_args, net, device):
    Approach = getattr(importlib.import_module(name='approach.' + method_name), 'Appr')
    Appr_ExemplarsDataset = Approach.exemplars_dataset_class()
    first_train_ds = trn_loader.dataset
    class_indices = first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger.logger)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(class_indices,
                                                                 **appr_exemplars_dataset_args)
    return Approach(net, device, **appr_kwargs)


def get_logger(full_exp_name):
    return Logger(MultiLogger(RESULT_PATH, full_exp_name, loggers=LOG, save_models=SAVE_MODELS))





def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic

