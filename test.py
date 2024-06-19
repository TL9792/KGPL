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

import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from network.model_KGPL import KGPL_model
from monai.data import decollate_batch 
import SimpleITK as sitk
from transformers import AutoTokenizer, AutoModel 
import torch.nn as nn
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    Invertd,
    SaveImaged,
) 

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline") 
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test", type=str, help="experiment name") 
parser.add_argument("--json_list", default="./dataset/test.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=1, type=int, help="data fold") 
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size") 
parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels") 
parser.add_argument("--out_channels", default=107, type=int, help="number of output channels")
parser.add_argument("--num_tokens", default=23, type=int, required=False,help="number of text tokens") 
parser.add_argument("--batch_size", default=1, type=int, required=False, help="number of batch size") 
parser.add_argument("--a_min", default=0.0, type=float, help="a_min in ScaleIntensityRanged") 
parser.add_argument("--a_max", default=3.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged") 
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged") 
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction") 
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", default=False, action="store_true", help="start distributed training")
parser.add_argument("--workers", default=2, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument(
    "--pretrained_dir",
    default="./outputs/",
    type=str,
    help="pretrained checkpoint directory",
) 


def main(): 
    args = parser.parse_args() 
    args.test_mode = True  
    test_loader, test_transform = get_loader(args) 

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

    from collections import OrderedDict  
    model_dict = torch.load(pretrained_pth) 
    new_state_dict = OrderedDict() 
    for k, v in model_dict["state_dict"].items(): 
        new_state_dict[k.replace("module.", "")] = v 
    model.load_state_dict(new_state_dict, strict=True) 
    model.eval() 
    model = model.cuda() 

    # post-process the pred
    post_transform = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_transform,
                orig_keys="image",
                nearest_interp=False,
            ),
        ] 
    ) 

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    ) 

    with torch.no_grad(): 
        for i, batch in enumerate(test_loader):  
            image = batch["image"].cuda() 
            orig_affine = batch["image_meta_dict"]["original_affine"][0].numpy() 
            case_name = batch["name"]  
            img_name = "pd-{}.nii.gz".format(model_dict['epoch']) 
            # save path 
            output_directory = "./outputs/prediction"
            if not os.path.exists(output_directory):  
                os.makedirs(output_directory) 

            ### select a BiomedCLIP model 
            biomedclip = AutoModel.from_pretrained('./text_model/')
            tokenizer = AutoTokenizer.from_pretrained('./text_model/') 

            word_embedding_list = [] 
            for j in range(len(case_name)): 
                sex=case_name[j].split('_')[-3] 
                age=case_name[j].split('_')[-2]
                state=case_name[j].split('_')[-1] 

                text = f"This is a brain magnetic resonance image acquired from a {sex} with {state} at {age} years old."

                # wanted word embedding in BERT 
                with torch.no_grad(): 
                    out = biomedclip(**tokenizer(text, return_tensors="pt"))
                    word_embeddings = out.last_hidden_state[0] # has no batch dimension
                word_embed = torch.zeros(args.num_tokens,768) 
                word_embed[:word_embeddings.shape[0],:] = word_embeddings
                word_embedding_list.append(word_embed) 
            word_embeddings_1 = torch.stack(word_embedding_list,dim=0) 
            word_embeddings_1 = word_embeddings_1.cuda() 

            print("Inference on case {}".format(case_name[0])) 
            batch["pred"] = model_inferer_test(image, word_embeddings_1) 
            # poseprocess the pred 
            batch = [post_transform(i) for i in decollate_batch(batch)] 
            seg_out = batch[0]["pred"] 
            prob = torch.sigmoid(seg_out) 
            seg = prob.detach().cpu().numpy() 
            seg = (seg > 0.5).astype(np.int8) 
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3])) 
            for i in range(107):
                seg_out[seg[i] == 1] = i
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), orig_affine), os.path.join(output_directory, img_name)) 
            # break

        print("Finished inference!") 


if __name__ == "__main__":
    main()
