'''
Train a zero-shot GAN using CLIP-based supervision.

Example commands:
    CUDA_VISIBLE_DEVICES=1 python train.py --size 1024 
                                           --batch 2 
                                           --n_sample 4 
                                           --output_dir /path/to/output/dir 
                                           --lr 0.002 
                                           --frozen_gen_ckpt /path/to/stylegan2-ffhq-config-f.pt 
                                           --iter 301 
                                           --source_class "photo" 
                                           --target_class "sketch" 
                                           --lambda_direction 1.0 
                                           --lambda_patch 0.0 
                                           --lambda_global 0.0 
                                           --lambda_texture 0.0 
                                           --lambda_manifold 0.0 
                                           --phase None 
                                           --auto_layer_k 0 
                                           --auto_layer_iters 0 
                                           --auto_layer_batch 8 
                                           --output_interval 50 
                                           --clip_models "ViT-B/32" "ViT-B/16" 
                                           --clip_model_weights 1.0 1.0 
                                           --mixing 0.0
                                           --save_interval 50
'''

import argparse
import os
import numpy as np

import torch

from tqdm import tqdm

from model.ZSSGAN_IDE3D import ZSSGAN
# from model.ZSSGAN import ZSSGAN
# from test.ZSSGAN_IDE3D import ZSSGAN

import shutil
import json
import pickle
import copy
import dnnlib

import snoop

from utils.file_utils import copytree, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise

from options.train_options import TrainOptions

#TODO convert these to proper args
SAVE_SRC = False
SAVE_DST = True


# from jojogan2d_stylize import _main_style
# from jojogan_util import *
import typing
import PIL.Image
import torchvision.transforms as transforms


class StyleImage(typing.NamedTuple):
    path: str
    aligned: torch.Tensor
    aligned2target = transforms.Compose(
        [
            # transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )



def _load_style_image(aligned_dir):

    style_aligned = PIL.Image.open(aligned_dir).convert('RGB')
    return StyleImage(
        path=aligned_dir,
        aligned=style_aligned,
    )



def load_style_images(style_images_aligned_dir):
    targets = []
    style_image = _load_style_image( style_images_aligned_dir)
    targets.append(StyleImage.aligned2target(style_image.aligned).to('cuda'))
    targets = torch.stack(targets, 0)

    return targets

def load_all_npy_w(flag,cam_path,w_path,target_img_dir):
    cam_pose=torch.from_numpy(np.load(cam_path)).to('cuda')
    if flag=='npy':
        w_pivot = torch.from_numpy(np.load(w_path)).to('cuda')
    if flag=='pt':
        w_pivot = torch.load(w_path).to('cuda')
    target = load_style_images(target_img_dir)


    return w_pivot,cam_pose,target

def load_npy_w(flag,w_path):

    cam_pose = torch.tensor([1,0,0,0, 0,-1,0,0, 0,0,-1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to('cuda').reshape(1, -1)
    if flag=='npy':
        w_pivot = torch.from_numpy(np.load(w_path)).to('cuda')
    if flag=='pt':
        w_pivot = torch.load(w_path).to('cuda')

    return w_pivot,cam_pose
#######################################################################


def load(path):
    ckpt = torch.load(path)
    # print("ckpt",ckpt)
    w = ckpt['w'].to('cuda')
    c = ckpt['c'].to("cuda")
    return w, c


# @snoop()
def train(args):
    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)
    z_dim =  512
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) # using original SG2 params. Not currently using r1 regularization, may need to change.

    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    g_optim_copy = torch.optim.Adam(
        net.generator_trainable_copy.parameters(),
        lr=2e-3, betas=(0, 0.99),
    )

    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "sample")
    ckpt_dir   = os.path.join(args.output_dir, "checkpoint")

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    np.random.seed(2)

    name=os.path.basename(args.input)
    w_pivot, cam_pose_target = load(args.input+f"/{name}.pt")
    target = load_style_images(args.input+f"/{name}.jpg")
    cam_pose= torch.tensor([1,0,0,0, 0,-1,0,0, 0,0,-1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to('cuda').reshape(1, -1)
    w_pivot=w_pivot.repeat(args.batch,1, 1)



    # Training loop
    fixed_z = torch.randn(args.n_sample, z_dim, device=device)
    for i in tqdm(range(args.iter)):

        net.train()
        sample_z = mixing_noise(args.batch, z_dim, args.mixing, device)
        z = torch.randn(args.batch, 512).to('cuda')
        ws = net.generator_frozen.style([z], cam_pose) # 得到 (n,14,512) 
        in_latent_nada = w_pivot.clone() 
        in_latent_jojo = w_pivot.clone() 
        id_swap_nada=[1,2,3,4,5,6,7,8,9,10,11,12,13]
        id_swap_jojo=[7,8,9,10,11,12,13]
        alpha_nada=0.2
        alpha_jojo=0
        in_latent_nada[:, id_swap_nada,:] = alpha_nada*w_pivot[:, id_swap_nada,:] + (1-alpha_nada)*ws[:, id_swap_nada,:] 
        in_latent_jojo[:, id_swap_jojo,:] = alpha_jojo*w_pivot[:, id_swap_jojo,:] + (1-alpha_jojo)*ws[:, id_swap_jojo,:] 

        [sampled_src, sampled_dst], loss = net(target,cam_pose_target,in_latent_nada,in_latent_jojo,i,args, sample_z)
        net.zero_grad()
        loss.backward()
        if args.type=="fusion":
            if i%2==0:
                g_optim_copy.step()
            else:
                g_optim.step()
        if args.type=="only_text":
            g_optim.step()
        if args.type=="only_img":
            g_optim_copy.step()
        tqdm.write(f"Clip loss: {loss}")

        if i % args.output_interval == 0:
            net.eval()
            with torch.no_grad():
                [sampled_src, sampled_dst], loss = net(target,cam_pose_target,in_latent_nada,in_latent_jojo,i,args, [fixed_z], truncation=args.sample_truncation)

                if args.crop_for_cars:
                    sampled_dst = sampled_dst[:, :, 64:448, :]

                grid_rows = int(args.n_sample ** 0.5)

                if SAVE_SRC:
                    save_images(sampled_src, sample_dir, "src", grid_rows, i)

                if SAVE_DST:
                    save_images(sampled_dst, sample_dir, "dst", grid_rows, i)


        if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):
            if args.type=="fusion":
                snapshot_eg3d_pkl = f'{ckpt_dir}/{str(i).zfill(6)}_eg3d_mixstyle.pkl'
                snapshot_eg3d_img_pkl = f'{ckpt_dir}/{str(i).zfill(6)}_eg3d_imgstyle.pkl'
                torch.save({"G_ema": copy.deepcopy(net.generator_trainable.generator).eval().requires_grad_(False).cpu()}, snapshot_eg3d_pkl)
                torch.save({"G_ema": copy.deepcopy(net.generator_trainable_copy.generator).eval().requires_grad_(False).cpu()}, snapshot_eg3d_img_pkl)


            if args.type=="only_text":
                snapshot_eg3d_text_pkl = f'{ckpt_dir}/{str(i).zfill(6)}_eg3d_textstyle.pkl'
                torch.save({"G_ema": copy.deepcopy(net.generator_trainable.generator).eval().requires_grad_(False).cpu()}, snapshot_eg3d_text_pkl)


            if args.type=="only_img":
                snapshot_eg3d_img_pkl = f'{ckpt_dir}/{str(i).zfill(6)}_eg3d_imgstyle.pkl'
                torch.save({"G_ema": copy.deepcopy(net.generator_trainable_copy.generator).eval().requires_grad_(False).cpu()}, snapshot_eg3d_img_pkl)
            

    for i in range(args.num_grid_outputs):
        net.eval()
        with torch.no_grad():
            sample_z = mixing_noise(16, z_dim, 0, device)
            [sampled_src, sampled_dst], _ = net(target,cam_pose_target,in_latent_nada,in_latent_jojo,i,args, sample_z, truncation=args.sample_truncation)

            if args.crop_for_cars:
                sampled_dst = sampled_dst[:, :, 64:448, :]

        save_paper_image_grid(sampled_dst, sample_dir, f"sampled_grid_{i}.png")
            


if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()

    # save snapshot of code / args before training.
    copytree("ZSSGAN/criteria/", os.path.join(args.output_dir, "code", "criteria"), )
    # shutil.copy2("ZSSGAN/model/ZSSGAN.py", os.path.join(args.output_dir, "code", "ZSSGAN.py"))
    
    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args)
    