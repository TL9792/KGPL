# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse 
import os 
from functools import partial 

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist 
import torch.multiprocessing as mp 
import torch.nn.parallel 
import torch.utils.data.distributed 
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR 
from trainer.trainer_structure import run_training  
from utils.data_utils import get_loader

from utils.inferers import sliding_window_inference 
from monai.losses import DiceLoss 
from monai.losses.focal_loss import FocalLoss 
from monai.metrics import DiceMetric 
from monai.networks.nets import SwinUNETR 
from github.network.model_KGPL import KGPL_model 
from monai.transforms import Activations, AsDiscrete, Compose 
from monai.utils.enums import MetricReduction  

from collections import OrderedDict 
from monai.networks.blocks import UnetOutBlock 

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline for BRATS Challenge") 
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint") 
parser.add_argument("--logdir", default="train", type=str, help="directory to save the tensorboard logs") 
parser.add_argument("--fold", default=1, type=int, required=False, help="data fold") 
parser.add_argument("--pretrained_model_name", default=None, type=str, help="pretrained model name")  
parser.add_argument("--data_dir", default="./dataset/", type=str, required=False,help="dataset directory")  
parser.add_argument("--json_list", default=".dataset/train.json", type=str, required=False, help="dataset json file") 
parser.add_argument("--save_checkpoint", action="store_true", default=True, required=False, help="save checkpoint during training")  
parser.add_argument("--max_epochs", default=1000, type=int, help="max number of training epochs") 
parser.add_argument("--batch_size", default=4, type=int, required=False, help="number of batch size") 
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size") 
parser.add_argument("--optim_lr", default=1e-4, type=float, required=False,help="optimization learning rate") 
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm") 
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight") 
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", default=True, required=False,help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, required=False,help="validation frequency")
parser.add_argument("--distributed", default=False, required=False, action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training") 
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training") 
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url") 
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend") 
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name") 
parser.add_argument("--workers", default=0, type=int, help="number of workers") 
parser.add_argument("--feature_size", default=48, type=int, required=False,help="feature size")
parser.add_argument("--in_channels", default=1, type=int, required=False,help="number of input channels")
parser.add_argument("--out_channels", default=107, type=int, required=False,help="number of output channels")
parser.add_argument("--num_tokens", default=23, type=int, required=False,help="number of text tokens")
parser.add_argument("--cache_dataset", action="store_true", help="use monai Dataset class") 
parser.add_argument("--a_min", default=0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=5000, type=float, help="a_max in ScaleIntensityRanged") 
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged") 
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged") 
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction") 
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction") 
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction") 
parser.add_argument("--roi_x", default=128, type=int, required=False,help="roi size in x direction") 
parser.add_argument("--roi_y", default=128, type=int, required=False,help="roi size in y direction") 
parser.add_argument("--roi_z", default=128, type=int, required=False,help="roi size in z direction") 
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate") 
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")  
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability") 
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability") 
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability") 
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, required=False,help="sliding window inference overlap") 
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, required=False,help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs") 
parser.add_argument("--resume_ckpt", action="store_true", default=True, help="resume training from pretrained checkpoint") 
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero") 
parser.add_argument("--use_checkpoint", default=True, required=False, action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, required=False, help="spatial dimension of input data") 
parser.add_argument(
    "--pretrained_dir",
    default="./pretrained_models",
    type=str, 
    help="pretrained checkpoint directory",
) 
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice") 
parser.add_argument("--device", default='cude', type=str, help="the used GPU") 


def main(): 
    args = parser.parse_args() 
    args.amp = not args.noamp 
    args.logdir = "./outputs/" 
    if args.distributed:  ## multi-GPU 
        args.ngpus_per_node = torch.cuda.device_count() 
        print("Found total gpus", args.ngpus_per_node) 
        args.world_size = args.ngpus_per_node * args.world_size 
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,)) 
    else: 
        main_worker(gpu=2, args=args) 


def main_worker(gpu, args): 
    if args.distributed: 
        torch.multiprocessing.set_start_method("fork", force=True) 
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)  
    args.gpu = gpu  
    if args.distributed:   
        args.rank = args.rank * args.ngpus_per_node + gpu 
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        ) 
    torch.cuda.set_device(args.gpu) 
    torch.backends.cudnn.benchmark = True  
    args.test_mode = False  
    loader,_ = get_loader(args)   
    if args.rank == 0:  
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs) 
    inf_size = [args.roi_x, args.roi_y, args.roi_z] 
    pretrained_dir = args.pretrained_dir 
    model_name = args.pretrained_model_name 
    pretrained_pth = os.path.join(pretrained_dir, model_name) 

    model = KGPL_model(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        num_tokens=args.num_tokens,
        batch=int(args.batch_size),
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
    ) 

    if args.squared_dice: 
        dice_loss = DiceLoss(
            to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        ) 
    else: 
        dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True) 
        focalLoss = FocalLoss(to_onehot_y=False) 
    post_sigmoid = Activations(sigmoid=True) 
    post_pred = AsDiscrete(argmax=False, logit_thresh=0.5) 
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True) 
    
    model_inferer = partial(
        sliding_window_inference, 
        roi_size=inf_size, 
        sw_batch_size=args.sw_batch_size, 
        predictor=model, 
        overlap=args.infer_overlap, 
    )    

    best_acc = 0 
    start_epoch = 0 

    # load the rspre-trained paramete 
    if args.resume_ckpt: 
        model_dict = torch.load(pretrained_pth)['state_dict']
        model_dict_again = OrderedDict() 
        for k, v in model_dict.items(): 
            model_dict_again[k.replace("module.", "")] = v 
        model.load_state_dict(model_dict_again, strict=False) 
        print("Using pretrained weights")

    ## load the checkpoint 
    if args.checkpoint is not None: 
        checkpoint = torch.load(args.checkpoint, map_location="cpu") 
        print('check:',checkpoint.keys())
        new_state_dict = OrderedDict() 
        for k, v in checkpoint["state_dict"].items():  
            new_state_dict[k.replace("backbone.", "")] = v 
        model.load_state_dict(new_state_dict, strict=False) 
        if "epoch" in checkpoint: 
            start_epoch = checkpoint["epoch"] 
        if "best_acc" in checkpoint:  
            best_acc = checkpoint["best_acc"] 
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc)) 

    # fix/refine the part of parameters 
    for name, parameters in model.named_parameters():  
        if ('swinViT' not in name):  
            parameters.requires_grad = True 
        else: 
            parameters.requires_grad = False 

    # cal the paramters of models 
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print("Total parameters count", pytorch_total_params) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    model = model.to(device)
    model  = nn.DataParallel(model)  

    if args.distributed: 
        torch.cuda.set_device(args.gpu) 
        if args.norm_name == "batch":  
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model.cuda(args.gpu) 
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    if args.optim_name == "adam":  
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw": 
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.optim_lr, weight_decay=args.reg_weight) 
    elif args.optim_name == "sgd": 
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )  
    else: 
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine": 
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        ) 
    elif args.lrschedule == "cosine_anneal": 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None: 
            scheduler.step(epoch=start_epoch) 
    else: 
        scheduler = None 

    if (args.checkpoint is not None) or (args.resume_ckpt): 
        checkpoint = torch.load(pretrained_pth) 
        if "optimizer" in checkpoint: 
            optimizer.load_state_dict(checkpoint(['optimizer']))
        if 'scheduler' in checkpoint: 
            scheduler.load_state_dict(checkpoint('scheduler'))

    semantic_classes = ["Dice_Val_GM", "Dice_Val_WM", "Dice_Val_CSF"] 

    accuracy = run_training(
        model=model, 
        train_loader=loader[0], 
        val_loader=loader[1], 
        optimizer=optimizer,
        loss_func_1=dice_loss, 
        loss_func_2=focalLoss,
        acc_func=dice_acc, 
        args=args, 
        model_inferer=model_inferer, 
        scheduler=scheduler,
        start_epoch=start_epoch, 
        post_sigmoid=post_sigmoid,
        post_pred=post_pred, 
        semantic_classes=semantic_classes, 
    )   
    return accuracy


if __name__ == "__main__":
    main() 
