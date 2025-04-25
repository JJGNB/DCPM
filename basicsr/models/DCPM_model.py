import numpy as np
import torch
from torch.nn import functional as F
from collections import OrderedDict
from os import path as osp
import torch.utils
from tqdm import tqdm
from basicsr.archs import build_network
from basicsr.archs.unet import UNetModelSwin
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models import lr_scheduler
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.convert2S012 import I2S012, Gen_CPFA, chunkbatch2dim, chunkdim2batch, init_CPDM
from basicsr.utils.registry import MODEL_REGISTRY
import torch.cuda.amp as amp
from basicsr.utils.script_util import create_gaussian_diffusion
import torchvision
@MODEL_REGISTRY.register()
class DDCPMRSModel(BaseModel):
    def __init__(self, opt):
        super(DDCPMRSModel, self).__init__(opt)
        # define network
        self.amp_scaler=amp.GradScaler()
        self.use_lq=opt['use_lq']
        self.use_vae=opt['use_vae']
        self.use_cpfa=opt['use_cpfa']
        self.use_unet=opt['use_unet']
        self.model=UNetModelSwin(**self.opt['Unet']['params'])
        self.model_to_device(self.model)
        # self.print_network(self.model)
        self.net_g=create_gaussian_diffusion(task_type="CPDM",**self.opt['Diff']['params'])
        # load pretrained models
        self.net_backbone=None
        load_path_diffusion=self.opt['path'].get('pretrain_network_g', None)
        if load_path_diffusion is not None:
            param_key_diffusion = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.model,load_path_diffusion, self.opt['path'].get('strict_load_g', True), param_key_diffusion)
        if self.is_train:
            self.init_training_settings()
    def init_training_settings(self):
        self.model.train()
        if self.use_vae:
            self.net_backbone.eval()
            for param in self.net_backbone.parameters():
                param.requires_grad = False
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = UNetModelSwin(**self.opt['Unet']['params']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_ema', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
                logger.info(f'load ema sucessfully!')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        self.setup_optimizers()
        self.setup_schedulers()
    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR',]:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'ExponentialLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.ExponentialLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingLR':
            for optimizer in self.optimizers:
                self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=train_opt['total_iter'] - train_opt['scheduler']['warmup_iterations'],
                    eta_min=train_opt['scheduler']['eta_min'],
                    ))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.model.named_parameters():
        # for k, v in self.net_backbone.decoder.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    def optimize_parameters(self, current_iter,mini_batch_size):
        self.optimizer_g.zero_grad()
        context=amp.autocast
        with context():
            if self.use_lq:
                self.lq=2*self.lq-1
            else:
                cpfa=Gen_CPFA(self.gt)
                self.lq=init_CPDM(cpfa)
                self.lq=2*self.lq-1
            self.gt=2*self.gt-1
            if self.use_cpfa:
                self.cpfa=2*self.cpfa-1
            else:
                self.cpfa=None
            bs=mini_batch_size
            t = torch.randint(0, self.opt['timestep'], (bs,), device=self.device).long()
            _,_,h,w=self.lq.shape
            noise=torch.randn(
                        size= (bs, 12,) + (h, ) * 2,
                        device=self.device,
                        )
            lq_3dim=chunkdim2batch(self.lq)
            gt_3dim=chunkdim2batch(self.gt)
            l_ddcpm,_,model_out=self.net_g.training_losses(self.model,gt_3dim,lq_3dim,t,bs=bs,first_stage_model=self.net_backbone,noise=noise,cpfa=self.cpfa)
            x_pred=model_out
            l_total = 0
            l_total+=l_ddcpm['mse']
            loss_dict = OrderedDict()
            loss_dict['l_ddcpm'] = l_ddcpm['mse']
            x_pred=x_pred*0.5+0.5
            self.gt=self.gt*0.5+0.5
            gt_S0,gt_DOLP=I2S012(self.gt,use_loss=True,clip=False,improve_contrast=False,save_img=True)
            pred_S0,pred_DOLP=I2S012(x_pred,use_loss=True,clip=False,improve_contrast=False,save_img=True,dim_type="N3HW")
            # pixel loss
            if self.cri_pix:
                l_pix = (self.cri_pix(gt_S0, pred_S0)+self.cri_pix(gt_DOLP, pred_DOLP))/2
                l_total += l_pix
                loss_dict['l_pix'] = l_pix
        self.amp_scaler.scale(l_total).backward()
        self.amp_scaler.step(self.optimizer_g)
        self.amp_scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results``
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            img_name=img_name.replace("_0","")
            img_name_0=img_name+"_0"
            img_name_45=img_name+"_45"
            img_name_90=img_name+"_90"
            img_name_135=img_name+"_135"
            img_name_S0=img_name+"_S0"
            img_name_DOLP=img_name+"_DOLP"
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            sr_img_0=visuals['result'][:,0:3,:,:]
            sr_img_45=visuals['result'][:,3:6,:,:]
            sr_img_90=visuals['result'][:,6:9,:,:]
            sr_img_135=visuals['result'][:,9:12,:,:]
            sr_img_S0,sr_img_DOLP=I2S012([sr_img_0,sr_img_45,sr_img_90,sr_img_135],save_img=True)
            sr_img_0 = tensor2img([sr_img_0])
            sr_img_45 = tensor2img([sr_img_45])
            sr_img_90 = tensor2img([sr_img_90])
            sr_img_135 = tensor2img([sr_img_135])
            sr_img_S0 = tensor2img([sr_img_S0])
            sr_img_DOLP = tensor2img([sr_img_DOLP])
            metric_data['img'] = [sr_img_0,sr_img_45,sr_img_90,sr_img_135,sr_img_S0,sr_img_DOLP]
            if 'gt' in visuals:
                gt_img_0=visuals['gt'][:,0:3,:,:]
                gt_img_45=visuals['gt'][:,3:6,:,:]
                gt_img_90=visuals['gt'][:,6:9,:,:]
                gt_img_135=visuals['gt'][:,9:12,:,:]
                gt_img_S0,gt_img_DOLP=I2S012([gt_img_0,gt_img_45,gt_img_90,gt_img_135],save_img=True)
                gt_img_0 = tensor2img([gt_img_0])
                gt_img_45 = tensor2img([gt_img_45])
                gt_img_90 = tensor2img([gt_img_90])
                gt_img_135 = tensor2img([gt_img_135])
                gt_img_S0 = tensor2img([gt_img_S0])
                gt_img_DOLP = tensor2img([gt_img_DOLP])
                metric_data['img2'] = [gt_img_0,gt_img_45,gt_img_90,gt_img_135,gt_img_S0,gt_img_DOLP]
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.cpfa
            del self.output
            torch.cuda.empty_cache()

            if save_img and (idx+1)%self.opt['val']['save_img_freq']==0:
                if self.opt['is_train']:
                    save_img_path_0 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name_0}_{current_iter}.png')
                    save_img_path_45 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name_45}_{current_iter}.png')
                    save_img_path_90 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name_90}_{current_iter}.png')
                    save_img_path_135 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name_135}_{current_iter}.png')
                    save_img_path_S0 = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name_S0}_{current_iter}.png')
                    save_img_path_DOLP = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name_DOLP}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img_0, save_img_path_0)
                imwrite(sr_img_45, save_img_path_45)
                imwrite(sr_img_90, save_img_path_90)
                imwrite(sr_img_135, save_img_path_135)
                imwrite(sr_img_S0, save_img_path_S0)
                imwrite(sr_img_DOLP, save_img_path_DOLP)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'cpfa' in data and self.use_cpfa:
            self.cpfa = data['cpfa'].to(self.device)
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.use_lq:
                    self.lq=2*self.lq-1
                else:
                    cpfa=Gen_CPFA(self.gt)
                    self.lq=init_CPDM(cpfa)
                    self.lq=2*self.lq-1
                if self.use_cpfa:
                    self.cpfa=2*self.cpfa-1
                else:
                    self.cpfa=None
                im_lq=chunkdim2batch(self.lq)
                num_iters = 0
                indices = np.linspace(
                    0,
                    self.net_g.num_timesteps,
                    self.net_g.num_timesteps if self.net_g.num_timesteps < 5 else 4,
                    endpoint=False,
                    dtype=np.int64,
                    ).tolist()
                if not (self.net_g.num_timesteps-1) in indices:
                    indices.append(self.net_g.num_timesteps-1)
                tt = torch.tensor(
                        [self.net_g.num_timesteps, ]*1,
                        dtype=torch.int64,
                        ).cuda()
                for sample in self.net_g.p_sample_loop_progressive(
                        y=im_lq,
                        cpfa=self.cpfa,
                        model=self.net_g_ema,
                        first_stage_model=self.net_backbone,
                        noise=None,
                        clip_denoised=True if self.net_backbone is None else False,
                        device=self.device,
                        progress=True,
                        ):
                    sample_decode = {}
                    if num_iters in indices:
                        for key, value in sample.items():
                            if key in ['sample', ]:
                                value=chunkdim2batch(value)
                                sample_decode[key] = self.net_g.decode_first_stage(
                                        value,
                                        self.net_backbone,
                                        ).clamp(-1.0, 1.0)
                        im_sr_progress = sample_decode['sample']
                    num_iters += 1
                    tt -= 1
                self.output=chunkbatch2dim(im_sr_progress)
                self.output=self.output*0.5+0.5
        else:
            self.model.eval()
            with torch.no_grad():
                if self.use_lq:
                    self.lq=2*self.lq-1
                else:
                    cpfa=Gen_CPFA(self.gt)
                    self.lq=init_CPDM(cpfa)
                    self.lq=2*self.lq-1
                if self.use_cpfa:
                    self.cpfa=2*self.cpfa-1
                else:
                    self.cpfa=None
                im_lq=chunkdim2batch(self.lq)
                num_iters = 0
                indices = np.linspace(
                    0,
                    self.net_g.num_timesteps,
                    self.net_g.num_timesteps if self.net_g.num_timesteps < 5 else 4,
                    endpoint=False,
                    dtype=np.int64,
                    ).tolist()
                if not (self.net_g.num_timesteps-1) in indices:
                    indices.append(self.net_g.num_timesteps-1)
                tt = torch.tensor(
                        [self.net_g.num_timesteps, ]*1,
                        dtype=torch.int64,
                        ).cuda()
                for sample in self.net_g.p_sample_loop_progressive(
                        y=im_lq,
                        cpfa=self.cpfa,
                        model=self.model,
                        first_stage_model=self.net_backbone,
                        noise=None,
                        clip_denoised=True if self.net_backbone is None else False,
                        device=self.device,
                        progress=True,
                        ):
                    sample_decode = {}
                    if num_iters in indices:
                        for key, value in sample.items():
                            if key in ['sample', ]:
                                value=chunkdim2batch(value)
                                sample_decode[key] = self.net_g.decode_first_stage(
                                        value,
                                        self.net_backbone,
                                        ).clamp(-1.0, 1.0)
                        im_sr_progress = sample_decode['sample']
                    num_iters += 1
                    tt -= 1
                self.output=chunkbatch2dim(im_sr_progress)
                self.output=self.output*0.5+0.5
            self.model.train()
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.model, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.model, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
