from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim,calculate_psnr_S0,calculate_ssim_S0,calculate_psnr_dolp,calculate_ssim_dolp
from .mi_vif import calculate_mi,calculate_viff,calculate_sd

__all__ = ['calculate_psnr_S0','calculate_psnr_dolp','calculate_psnr','calculate_ssim_S0','calculate_ssim_dolp', 'calculate_ssim', 'calculate_niqe','calculate_mi','calculate_viff','calculate_sd']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
