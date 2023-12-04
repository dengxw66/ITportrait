import sys
import os
from tabnanny import check
sys.path.insert(0, os.path.abspath('.'))


import torch
import torchvision.transforms as transforms
import json
import numpy as np
import copy
import pickle

import snoop

from copy import deepcopy


from functools import partial

from ZSSGAN.criteria.clip_loss import CLIPLoss_middle       
from ZSSGAN.criteria.id_loss import IDLoss   
import ZSSGAN.legacy as legacy

import dnnlib
import math
import random
from ZSSGAN.camera_utils import LookAtPoseSampler
import lpips
import torch.nn.functional as F


import typing
import PIL.Image
import torchvision.transforms as transforms


class StyleImage(typing.NamedTuple):
    path: str
    aligned: torch.Tensor

    aligned2target = transforms.Compose(
        [
            # transforms.Resize((512,512)),
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




def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics


def test_eg3d():
    cam_pose= torch.tensor([1.0000e+00, -6.4913e-09,  4.6791e-08, -1.1706e-07,  0.0000e+00,
         -9.9051e-01, -1.3741e-01,  3.4377e-01,  4.7239e-08,  1.3741e-01,
         -9.9051e-01,  2.6780e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          1.0000e+00,  4.2634e+00,  0.0000e+00,  5.0000e-01,  0.0000e+00,
          4.2634e+00,  5.0000e-01,  0.0000e+00,  0.0000e+00,  1.0000e+00]).float().to('cuda').reshape(1, -1)


    network_pkl='/data1/dxw_data/code/github/iccv2023/jojogan_nada/pretrained/ffhqrebalanced512-128.pkl'
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    # with dnnlib.util.open_url(network_pkl) as f:
    #     import legacy
    #     original_generator = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device) # type: ignore

    with open(network_pkl, 'rb') as f:
        original_generator = pickle.load(f)['G_ema'].cuda() 

    # 加载模型有问题。看起来对的
    # 其实不对劲。但是我的加载模型应该也没有问题啊

    generator = deepcopy(original_generator)


    z = torch.randn(1, 512).to('cuda') # z 就是随机数，也ok   # cam 没有问题
    # 可能的问题是ws和
    ws = generator.mapping(z, cam_pose)

    # 生成图片：
    # cam_pose 肯定正确
    # w也是对的
    w_pivot = torch.from_numpy(np.load('/data1/dxw_data/code/github/iccv2023/jojogan_3d/w_pose_npy/w_jojogan_200_jinkes2.npy')).to('cuda')
    trainable_img_jojogan= generator.synthesis(w_pivot, cam_pose, noise_mode='const')['image']# trainable_img_jojogan.shape = (1, 3, 512, 512)

    # 测试保存：
    from torchvision.utils import save_image
    path_jojo1='/data1/dxw_data/code/ide3d-nada-main/output/eg3d_nada_output/test_jojo_zssgan.png'
    save_image(trainable_img_jojogan,path_jojo1,normalize=True)

    return trainable_img_jojogan 

def jojogan_img_loss_eg3d(img,target_img):

    lpips_fn = lpips.LPIPS(net='vgg').to('cuda') # 用于度量两个图片的相似度
    loss = lpips_fn(F.interpolate(img, size=(256,256), mode='area'), F.interpolate(target_img, size=(256,256), mode='area')).mean()

    from torchvision.utils import save_image
    target_path1='/data1/dxw_data/code/ide3d-nada-main/output/view_multi/src_jojo.jpg'
    # target_path2='/data1/dxw_data/code/ide3d-nada-main/sh/target_jojo.jpg'
    save_image(img,target_path1,normalize=True)
    # save_image(target_img,target_path2,normalize=True)
    # print("jojoloss",loss)

    return loss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class EG3DGenerator(torch.nn.Module):

    def __init__(self, checkpoint_path):
        super(EG3DGenerator, self).__init__()
        with open(checkpoint_path, 'rb') as f:
            if checkpoint_path.endswith('.pt'):
                # 这里是为了Art3D准备的
                self.generator = torch.load(f)# .cuda()
                print("self.generator",self.generator)
            else: # 对于ITportrait而言的
                self.generator = pickle.load(f)['G_ema'].cuda() 
                print("self.generator ",self.generator)


    # @snoop()
    def get_all_layers(self): # 问题很多，核心
        layers = []
        # 原始：for child in self.generator.synthesis.children():
        # for child in self.generator.backbone.children():
        for child in self.generator.children(): 
            # 自动加上generator所有的成员，所以应该是正确的，没有毛病
            # 可以选择只优化backbone
            layers += list(child.children())
            # return list(self.generator.synthesis.children())
        return layers

    def trainable_params(self):
        params = []
        for layer in self.get_training_layers():
            params.extend(layer.parameters())

        return params

    def get_training_layers(self, phase=None):
        return self.get_all_layers()

    def freeze_layers(self, layer_list=None):

        if layer_list is None:
            self.freeze_layers(self.generator.children())
        else:
            for layer in layer_list:
                requires_grad(layer, False) # 全部冻上

    def unfreeze_layers(self, layer_list=None):

        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True) # 全部解冻
                
    # @snoop()
    def style(self, z_codes, c, truncation=0.7):
        # 输入：z_codes(batch，512)，c 摄像机位姿
        # 输出：w（batch，18,512）或者w（batch，14,512）
        # 原始的：return self.generator.mapping(z_codes[0], c.repeat(z_codes[0].shape[0], 1), truncation_psi=truncation, truncation_cutoff=None)
        return self.generator.mapping(z_codes[0], c.repeat(z_codes[0].shape[0], 1))

    # @snoop()
    def forward(self, styles, c=None, truncation=None, randomize_noise=True): # unused args for compatibility with SG2 interface
        # 输入：w（batch，18,512）或者w（batch，14,512），c 摄像机位姿
        # 输出：img.shape = (4, 3, 512, 512)
        # noise_mode = 'random' if randomize_noise else 'const'
        # 原始的：return self.generator.synthesis(styles, c, noise_mode=noise_mode, force_fp32=True), None
        # styles.shape = (8, 14, 512),c.shape = (8, 25) 就是值太大了，如果少一些就ok
        # return self.generator.synthesis(styles, c, noise_mode='const'), None

        return self.generator.synthesis(styles, c, noise_mode='const')


class ZSSGAN(torch.nn.Module):
    def __init__(self, args):
        super(ZSSGAN, self).__init__()

        self.args = args

        self.device = 'cuda:0'

        self.generator_frozen = EG3DGenerator(args.frozen_gen_ckpt)
        self.generator_trainable = EG3DGenerator(args.train_gen_ckpt)
        self.generator_trainable_copy = EG3DGenerator(args.train_gen_ckpt)



        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()


        self.generator_trainable.freeze_layers()
        self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase))
        self.generator_trainable.train()
        
        self.generator_trainable_copy.freeze_layers()
        self.generator_trainable_copy.unfreeze_layers(self.generator_trainable_copy.get_training_layers(args.phase))
        self.generator_trainable_copy.train()


        self.c_front = torch.tensor([1,0,0,0, 0,-1,0,0, 0,0,-1,2.7, 0,0,0,1, 4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).float().to('cuda').reshape(1, -1)
        
        self.clip_loss_models_middle = {model_name: CLIPLoss_middle(self.device, 
                                                      lambda_direction=args.lambda_direction, 
                                                      lambda_patch=args.lambda_patch, 
                                                      lambda_global=args.lambda_global, 
                                                      lambda_manifold=args.lambda_manifold, 
                                                      lambda_texture=args.lambda_texture,
                                                      clip_model=model_name) 
                                for model_name in args.clip_models}

        self.clip_model_weights = {model_name: weight for model_name, weight in zip(args.clip_models, args.clip_model_weights)}

        self.mse_loss  = torch.nn.MSELoss()

        self.source_class = args.source_class
        self.target_class = args.target_class

        self.auto_layer_k     = args.auto_layer_k
        self.auto_layer_iters = args.auto_layer_iters


    def determine_opt_layers(self,flag):

        if flag=='generator_trainable':
            all_layers = list(self.generator_trainable.get_all_layers()) 
        if flag=='generator_trainable_copy':
            all_layers = list(self.generator_trainable_copy.get_all_layers()) 

        conv_inds = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 23, 24, 26, 27]
        rgb_inds = [1, 4, 7, 10, 13, 16, 19, 25, 28]


        need_inds=[1,3,4] 

        conv_layers = []
        rgb_layers = []
        nerf_layers = []
        need_layers=[]

        for i, layer in enumerate(all_layers):

            if i in need_inds:
                need_layers.append(layer)
            elif i in conv_inds:
                conv_layers.append(layer)
            elif i in rgb_inds:
                rgb_layers.append(layer)
            else:
                nerf_layers.append(layer)

        chosen_layers = need_layers
        
      

        return chosen_layers

   
    def forward(
        self,
        target_img,
        cam_pose_target,
        w_get_nada,
        w_get_jojo,
        epoch,
        args,
        styles,
        labels=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        c = labels if labels is not None else self.c_front


        # if args.view=='multi':
        # print("styles[0].shape",styles[0].shape) # torch.Size([2, 512])
        with torch.no_grad():
            if input_is_latent:
                w_styles = styles
            else:
                w_styles = self.generator_frozen.style(styles, c) # w_styles.shape = (2, 14, 512)

        if args.type=="fusion":
            if epoch%2==0:
                if self.training and self.auto_layer_iters > 0:
                    self.generator_trainable_copy.unfreeze_layers()
                    train_layers_copy = self.determine_opt_layers('generator_trainable_copy')
                    if not isinstance(train_layers_copy, list): 
                        train_layers_copy = [train_layers_copy]
                    self.generator_trainable_copy.freeze_layers() 
                    self.generator_trainable_copy.unfreeze_layers(train_layers_copy) 

                print("-----------------------stage1-----------------------")
                trainable_img_multiview_jojo = self.generator_trainable_copy(w_get_jojo[0].unsqueeze(0), c=cam_pose_target, truncation=truncation, randomize_noise=randomize_noise)['image']
                jojoloss=jojogan_img_loss_eg3d(trainable_img_multiview_jojo,target_img)

                finalloss=jojoloss 
                return [trainable_img_multiview_jojo, trainable_img_multiview_jojo], finalloss

            else:
                if self.training and self.auto_layer_iters > 0:
                    self.generator_trainable.unfreeze_layers()
                    train_layers = self.determine_opt_layers('generator_trainable')
                    if not isinstance(train_layers, list): 
                        train_layers = [train_layers]
                    self.generator_trainable.freeze_layers() 
                    self.generator_trainable.unfreeze_layers(train_layers) 

                print("-----------------------mix，stage2-----------------------")

                center_pose=torch.tensor( [[ 1.0000e+00, -1.9107e-15, -4.3711e-08, -1.1802e-07,  0.0000e+00,
                    -1.0000e+00, -4.3711e-08, -1.1802e-07,  4.3711e-08,  4.3711e-08,
                    -1.0000e+00,  2.7000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                    1.0000e+00,  4.2647e+00,  0.0000e+00,  5.0000e-01,  0.0000e+00,
                    4.2647e+00,  5.0000e-01,  0.0000e+00,  0.0000e+00,  1.0000e+00],]).cuda() 
                pose_center=center_pose[0].unsqueeze(0)  
                fov_deg=18
                intrinsics = FOV_to_intrinsics(fov_deg, device='cuda')
                cam_pivot=torch.tensor([0, 0, 0.2]).to('cuda')
                angle_y=[0,0,0,0,0]
                angle_p=[0,0,0,0,0]
                angle_y[0]=(random.randrange(-6,-3))/10 # (-7,-3)
                angle_p[0]=(random.randrange(-6,3))/10 # (-8,4)
                angle_y[1]=(random.randrange(-3,7))/10 # (-7,7)
                angle_p[1]=(random.randrange(-6,3))/10 # (-8,4)
                angle_y[2]=(random.randrange(3,6))/10 # (3,7)
                angle_p[2]=(random.randrange(-6,3))/10 # (-8,4)
                angle_y[3]=(random.randrange(-3,7))/10 # (3,7)
                angle_p[3]=(random.randrange(-6,3))/10 # (-8,4)
                angle_y[4]=(random.randrange(-3,7))/10 # (3,7)
                angle_p[4]=(random.randrange(-6,3))/10 # (-8,4)

                aug_multi_view_imgs_train = []
                aug_multi_view_imgs_train_fix = []
                aug_multi_view_imgs_frozen = []
                for i in range(2): 
                    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y[i], np.pi/2 + angle_p[i], cam_pivot, radius=2.7, device='cuda')
                    camera_params_t = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                    trainable_img_multiview = self.generator_trainable(w_get_nada, c=camera_params_t.repeat(w_get_nada.shape[0],1), truncation=truncation, randomize_noise=randomize_noise)['image']
                    with torch.no_grad(): 
                        fix_img_multiview = self.generator_trainable_copy(w_get_nada, c=camera_params_t.repeat(w_get_nada.shape[0],1), truncation=truncation, randomize_noise=randomize_noise)['image'] 
                        frozen_img_multiview = self.generator_frozen(w_get_nada, c=pose_center.repeat(w_get_nada.shape[0],1), truncation=truncation, randomize_noise=randomize_noise)['image'] 
                    aug_multi_view_imgs_train_fix.append(fix_img_multiview)
                    aug_multi_view_imgs_train.append(trainable_img_multiview)
                    aug_multi_view_imgs_frozen.append(frozen_img_multiview)
                aug_multi_view_imgs_train_fix = torch.cat(aug_multi_view_imgs_train_fix,dim=0)
                aug_multi_view_imgs_train = torch.cat(aug_multi_view_imgs_train,dim=0) 
                aug_multi_view_imgs_frozen = torch.cat(aug_multi_view_imgs_frozen,dim=0) 
                
                clip_loss = torch.sum(torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models_middle[model_name](aug_multi_view_imgs_train_fix,epoch,args,aug_multi_view_imgs_frozen, self.source_class, aug_multi_view_imgs_train, self.target_class) for model_name in self.clip_model_weights.keys()]))


                # from torchvision.utils import save_image
                # # test：
                # trainable_img_dir='/data1/dxw_data/code/github/ide3d-nada-main/output/trainable_img.jpg'
                # frozen_img_dir='/data1/dxw_data/code/github/ide3d-nada-main/output/frozen_img.jpg'
                # fix_target_img_dir='/data1/dxw_data/code/github/ide3d-nada-main/output/fix_target_img.jpg'
                # save_image(aug_multi_view_imgs_train,trainable_img_dir,normalize=True)
                # save_image(aug_multi_view_imgs_frozen,frozen_img_dir,normalize=True)
                # save_image(aug_multi_view_imgs_train_fix,fix_target_img_dir,normalize=True)

                finalloss=clip_loss
                return [aug_multi_view_imgs_frozen, aug_multi_view_imgs_train], finalloss

        if args.type=="only_text":
            if self.training and self.auto_layer_iters > 0:
                self.generator_trainable.unfreeze_layers()
                train_layers_copy = self.determine_opt_layers('generator_trainable')
                if not isinstance(train_layers_copy, list): 
                    train_layers_copy = [train_layers_copy]
                self.generator_trainable.freeze_layers() 
                self.generator_trainable.unfreeze_layers(train_layers_copy) 

            # with torch.no_grad():
            #     frozen_img = self.generator_frozen(w_get_nada, c=c.repeat(w_get_nada.shape[0], 1), truncation=truncation, randomize_noise=randomize_noise)['image']
            # trainable_img = self.generator_trainable(w_get_nada, c=c.repeat(w_get_nada.shape[0], 1), truncation=truncation, randomize_noise=randomize_noise)['image']
            center_pose=torch.tensor( [[ 1.0000e+00, -1.9107e-15, -4.3711e-08, -1.1802e-07,  0.0000e+00,
                    -1.0000e+00, -4.3711e-08, -1.1802e-07,  4.3711e-08,  4.3711e-08,
                    -1.0000e+00,  2.7000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                    1.0000e+00,  4.2647e+00,  0.0000e+00,  5.0000e-01,  0.0000e+00,
                    4.2647e+00,  5.0000e-01,  0.0000e+00,  0.0000e+00,  1.0000e+00],]).cuda() 
            pose_center=center_pose[0].unsqueeze(0)  
            fov_deg=18
            intrinsics = FOV_to_intrinsics(fov_deg, device='cuda')
            cam_pivot=torch.tensor([0, 0, 0.2]).to('cuda')
            angle_y=[0,0,0,0,0]
            angle_p=[0,0,0,0,0]
            angle_y[0]=(random.randrange(-6,-3))/10 # (-7,-3)
            angle_p[0]=(random.randrange(-6,3))/10 # (-8,4)
            angle_y[1]=(random.randrange(-3,7))/10 # (-7,7)
            angle_p[1]=(random.randrange(-6,3))/10 # (-8,4)
            angle_y[2]=(random.randrange(3,6))/10 # (3,7)
            angle_p[2]=(random.randrange(-6,3))/10 # (-8,4)
            angle_y[3]=(random.randrange(-3,7))/10 # (3,7)
            angle_p[3]=(random.randrange(-6,3))/10 # (-8,4)
            angle_y[4]=(random.randrange(-3,7))/10 # (3,7)
            angle_p[4]=(random.randrange(-6,3))/10 # (-8,4)

            aug_multi_view_imgs_train = []
            aug_multi_view_imgs_train_fix = []
            aug_multi_view_imgs_frozen = []
            for i in range(2): 
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y[i], np.pi/2 + angle_p[i], cam_pivot, radius=2.7, device='cuda')
                camera_params_t = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                trainable_img_multiview = self.generator_trainable(w_get_nada, c=camera_params_t.repeat(w_get_nada.shape[0],1), truncation=truncation, randomize_noise=randomize_noise)['image']
                with torch.no_grad(): 
                    fix_img_multiview = self.generator_trainable_copy(w_get_nada, c=camera_params_t.repeat(w_get_nada.shape[0],1), truncation=truncation, randomize_noise=randomize_noise)['image'] 
                    frozen_img_multiview = self.generator_frozen(w_get_nada, c=pose_center.repeat(w_get_nada.shape[0],1), truncation=truncation, randomize_noise=randomize_noise)['image'] 
                aug_multi_view_imgs_train_fix.append(fix_img_multiview)
                aug_multi_view_imgs_train.append(trainable_img_multiview)
                aug_multi_view_imgs_frozen.append(frozen_img_multiview)
            aug_multi_view_imgs_train_fix = torch.cat(aug_multi_view_imgs_train_fix,dim=0)
            aug_multi_view_imgs_train = torch.cat(aug_multi_view_imgs_train,dim=0) 
            aug_multi_view_imgs_frozen = torch.cat(aug_multi_view_imgs_frozen,dim=0) 
            
            clip_loss = torch.sum(torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models_middle[model_name](aug_multi_view_imgs_frozen,epoch,args,aug_multi_view_imgs_frozen, self.source_class, aug_multi_view_imgs_train, self.target_class) for model_name in self.clip_model_weights.keys()]))
            finalloss=clip_loss

            # from torchvision.utils import save_image
            # # test：
            # trainable_img_dir='/data1/dxw_data/code/github/ide3d-nada-main/output/trainable_img.jpg'
            # frozen_img_dir='/data1/dxw_data/code/github/ide3d-nada-main/output/frozen_img.jpg'
            # fix_target_img_dir='/data1/dxw_data/code/github/ide3d-nada-main/output/fix_target_img.jpg'
            # save_image(aug_multi_view_imgs_train,trainable_img_dir,normalize=True)
            # save_image(aug_multi_view_imgs_frozen,frozen_img_dir,normalize=True)
            # save_image(aug_multi_view_imgs_train_fix,fix_target_img_dir,normalize=True)

            return [aug_multi_view_imgs_frozen, aug_multi_view_imgs_train], finalloss

        if args.type=="only_img":
            if self.training and self.auto_layer_iters > 0:
                self.generator_trainable_copy.unfreeze_layers()
                train_layers_copy = self.determine_opt_layers('generator_trainable_copy')
                if not isinstance(train_layers_copy, list): 
                    train_layers_copy = [train_layers_copy]
                self.generator_trainable_copy.freeze_layers() 
                self.generator_trainable_copy.unfreeze_layers(train_layers_copy) 

            print("-----------------------stage1-----------------------")
            trainable_img_multiview_jojo = self.generator_trainable_copy(w_get_jojo[0].unsqueeze(0), c=cam_pose_target, truncation=truncation, randomize_noise=randomize_noise)['image']
            jojoloss=jojogan_img_loss_eg3d(trainable_img_multiview_jojo,target_img)

            finalloss=jojoloss 
            from torchvision.utils import save_image
            # test：
            
            return [trainable_img_multiview_jojo, trainable_img_multiview_jojo], finalloss

    def pivot(self):
        par_frozen = dict(self.generator_frozen.named_parameters())
        par_train  = dict(self.generator_trainable.named_parameters())

        for k in par_frozen.keys():
            par_frozen[k] = par_train[k]


if __name__ == "__main__":
    print("load_ eg3d _generator ...") # 这里加载预训练的模型
    # 这里变成使用eg3d
    # original_generator = load_original_generator(args.pretrained, args.latent_dim)
    network_pkl='/data1/dxw_data/code/github/iccv2023/jojogan_3d/pretrained/ffhqrebalanced512-128.pkl'
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        # 这里得到G的预训练模型
        import legacy
        original_generator = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device) # type: ignore
    print("original_generator",original_generator)

    generator = deepcopy(original_generator)

    
