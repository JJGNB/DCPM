import os
import numpy as np
import torch
import torchvision
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torchvision import transforms
from basicsr.archs.unet import UNetModelSwin
from basicsr.models.diffusion_main import  UnetRes
from PIL import Image
from os import path as osp
from basicsr.utils.convert2S012 import I2S012, Gen_CPFA, init_CPDM
from basicsr.utils.misc import set_random_seed
from basicsr.utils.options import parse_options
from basicsr.utils.script_util import ImageSpliterTh, create_gaussian_diffusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from contextlib import contextmanager
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
def fast_inference(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id']
    use_lq=opt['use_lq']
    real_cpdm=opt['real_cpdm']
    chop_size=opt['datasets']['chop_size']
    improve_contrast=opt['datasets']['save_path']['improve_contrast']
    if real_cpdm:
        gt_path_cpfa=opt['datasets']['val']['dataroot_gt_cpfa']
        output_path_0=opt['datasets']['save_path']['img_0']
        output_path_45=opt['datasets']['save_path']['img_45']
        output_path_90=opt['datasets']['save_path']['img_90']
        output_path_135=opt['datasets']['save_path']['img_135']
        output_path_S0=opt['datasets']['save_path']['img_S0']
        output_path_DLOP=opt['datasets']['save_path']['img_DOLP']
    else:
        input_path_0=opt['datasets']['val']['dataroot_lq_0']
        input_path_45=opt['datasets']['val']['dataroot_lq_45']
        input_path_90=opt['datasets']['val']['dataroot_lq_90']
        input_path_135=opt['datasets']['val']['dataroot_lq_135']
        input_path_cpfa=opt['datasets']['val']['dataroot_lq_cpfa']
        gt_path_0=opt['datasets']['val']['dataroot_gt_0']
        gt_path_45=opt['datasets']['val']['dataroot_gt_45']
        gt_path_90=opt['datasets']['val']['dataroot_gt_90']
        gt_path_135=opt['datasets']['val']['dataroot_gt_135']
        output_path_0=opt['datasets']['save_path']['img_0']
        output_path_45=opt['datasets']['save_path']['img_45']
        output_path_90=opt['datasets']['save_path']['img_90']
        output_path_135=opt['datasets']['save_path']['img_135']
        output_path_S0=opt['datasets']['save_path']['img_S0']
        output_path_DLOP=opt['datasets']['save_path']['img_DOLP']
    if not os.path.exists(output_path_S0):
        os.makedirs(output_path_0)
        os.makedirs(output_path_45)
        os.makedirs(output_path_90)
        os.makedirs(output_path_135)
        os.makedirs(output_path_S0)
        os.makedirs(output_path_DLOP)
    set_random_seed(opt['seed'])
    backbone=None
    model=UNetModelSwin(**opt['Unet']['params'])
    weight_diffusion=opt['path']['pretrain_diffsuion_unet']
    model.load_state_dict(torch.load(weight_diffusion)['params_ema'])
    param_cnt = sum(p.numel() for p in model.parameters())
    print("#Param.", param_cnt/1e6, "M")
    model.to("cuda:{0}".format(opt['gpu_id']))
    model.eval()
    net_g=create_gaussian_diffusion(task_type='CPDM',**opt['Diff']['params'])
    if real_cpdm:
        name_list=os.listdir(gt_path_cpfa)
    else:
        name_list=os.listdir(input_path_0)
    tfs=transforms.Compose([
        transforms.CenterCrop([chop_size,chop_size]),
    transforms.ToTensor()
    ])

    tfs_cpfa=transforms.Compose([
        transforms.Grayscale(1),
    transforms.ToTensor()
    ])
    index=0
    for name in name_list:
        index+=1
        if real_cpdm:
            base_name=name.replace('.png','')
            cpfa=tfs(Image.open(os.path.join(gt_path_cpfa,name))).unsqueeze(0)
            lq=init_CPDM(cpfa[:,0:1,:,:]).to("cuda:{0}".format(opt['gpu_id']))
        else:
            base_name=name.replace('_0.png','')
            gt_0=tfs(Image.open(os.path.join(gt_path_0,base_name+'_0.png'))).unsqueeze(0)
            gt_45=tfs(Image.open(os.path.join(gt_path_45,base_name+'_45.png'))).unsqueeze(0)
            gt_90=tfs(Image.open(os.path.join(gt_path_90,base_name+'_90.png'))).unsqueeze(0)
            gt_135=tfs(Image.open(os.path.join(gt_path_135,base_name+'_135.png'))).unsqueeze(0)
            gt=torch.concat([gt_0,gt_45,gt_90,gt_135],dim=1).to("cuda:{0}".format(opt['gpu_id']))
            if use_lq:
                lq_0=tfs(Image.open(os.path.join(input_path_0,base_name+'_0.png'))).unsqueeze(0)
                lq_45=tfs(Image.open(os.path.join(input_path_45,base_name+'_45.png'))).unsqueeze(0)
                lq_90=tfs(Image.open(os.path.join(input_path_90,base_name+'_90.png'))).unsqueeze(0)
                lq_135=tfs(Image.open(os.path.join(input_path_135,base_name+'_135.png'))).unsqueeze(0)
                lq_cpfa=tfs_cpfa(Image.open(os.path.join(input_path_cpfa,base_name+'_135.png'))).unsqueeze(0).to("cuda:{0}".format(opt['gpu_id']))
                lq=torch.concat([lq_0,lq_45,lq_90,lq_135],dim=1).to("cuda:{0}".format(opt['gpu_id']))
            else:
                cpfa=Gen_CPFA(gt)
                lq=init_CPDM(cpfa)
        lq_copy=lq.clone()
        context = torch.cuda.amp.autocast
        with context():
            lq_0=lq_copy[:,0:3,:,:]
            lq_45=lq_copy[:,3:6,:,:]
            lq_90=lq_copy[:,6:9,:,:]
            lq_135=lq_copy[:,9:12,:,:]
            lq=2*lq-1
            with context():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                im_sr_tensor =net_g.p_sample_loop(
                    y=lq,
                    model=model,
                    first_stage_model=backbone,
                    noise=None,
                    clip_denoised=True if backbone is None else False,
                    device=None,
                    progress=True,
                    )     # 1 x c x h x w, [-1, 1]
                end_event.record()
                torch.cuda.synchronize()
                # 计算时间
                inference_time = start_event.elapsed_time(end_event)
                print(f"模型推理时间: {inference_time:.3f} ms")
            im_sr_tensor = 0.5*im_sr_tensor+0.5
            output_0=im_sr_tensor[:,0:3,:,:]
            output_45=im_sr_tensor[:,3:6,:,:]
            output_90=im_sr_tensor[:,6:9,:,:]
            output_135=im_sr_tensor[:,9:12,:,:]
            output_S0,output_DOLP=I2S012([output_0,output_45,output_90,output_135],save_img=True,improve_contrast=improve_contrast)
        torchvision.utils.save_image(output_0,os.path.join(output_path_0,base_name+'_0.png'))
        torchvision.utils.save_image(output_45,os.path.join(output_path_45,base_name+'_45.png'))
        torchvision.utils.save_image(output_90,os.path.join(output_path_90,base_name+'_90.png'))
        torchvision.utils.save_image(output_135,os.path.join(output_path_135,base_name+'_135.png'))
        torchvision.utils.save_image(output_S0,os.path.join(output_path_S0,base_name+'_S0.png'))
        torchvision.utils.save_image(output_DOLP,os.path.join(output_path_DLOP,base_name+'_DOLP.png'))
        print(str(index)+"done!")
if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    fast_inference(root_path)








