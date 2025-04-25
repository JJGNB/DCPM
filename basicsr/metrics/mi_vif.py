import cv2
import numpy as np
import torch
import math
import torch.nn.functional as F
from scipy.signal import convolve2d
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY
@METRIC_REGISTRY.register()
def calculate_sd(fuse_img, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    fuse_img=reorder_image(fuse_img, input_order=input_order)

    if crop_border != 0:
        fuse_img = fuse_img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        fuse_img=to_y_channel(fuse_img)[:,:,0]

    fuse_img = fuse_img.astype(np.float32)
    SD=contrast(torch.from_numpy(fuse_img)/255)
    return SD.item()
def contrast(input):
    input=input.unsqueeze(0)
    input=input.unsqueeze(0)
    N,C,H,W=input.shape
    input_m = torch.mean(input, dim=[2, 3], keepdim=True)
    ct = torch.sqrt(torch.sum((input - input_m) ** 2, dim=[2, 3], keepdim=True)/(H*W))
    return ct
@METRIC_REGISTRY.register()
def calculate_mi(fuse_img,img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    if fuse_img.shape != img2.shape:
        img2=img2[:,:,np.newaxis]
        img2=np.concatenate([img2,img2,img2],axis=2)
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    fuse_img=reorder_image(fuse_img, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        fuse_img = fuse_img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)[:,:,0]
        img2 = to_y_channel(img2)[:,:,0]
        fuse_img=to_y_channel(fuse_img)[:,:,0]

    img = img.astype(np.int32)
    img2 = img2.astype(np.int32)
    fuse_img = fuse_img.astype(np.int32)
    MI= Hab(img, fuse_img, 256)+Hab(img2, fuse_img, 256)
    return MI
@METRIC_REGISTRY.register()
def calculate_viff(fuse_img,img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    if fuse_img.shape != img2.shape:
        img2=img2[:,:,np.newaxis]
        img2=np.concatenate([img2,img2,img2],axis=2)
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    fuse_img=reorder_image(fuse_img, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        fuse_img = fuse_img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)[:,:,0]
        img2 = to_y_channel(img2)[:,:,0]
        fuse_img=to_y_channel(fuse_img)[:,:,0]

    img = img.astype(np.float32)
    img2 = img2.astype(np.float32)
    fuse_img = fuse_img.astype(np.float32)
    VIFF=vifp_mscale(img,fuse_img)+vifp_mscale(img2,fuse_img)
    return VIFF
def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = fspecial_gaussian((N, N), N / 5)

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist * dist, win, mode='valid') - mu2_sq
        sigma12 = convolve2d(ref * dist, win, mode='valid') - mu1_mu2
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10

        num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    vifp = num / den
    return vifp
def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
def Hab(im1, im2, gray_level):
    hang, lie = im1.shape
    count = hang * lie
    N = gray_level
    h = np.zeros((N, N))
    for i in range(hang):
        for j in range(lie):
            h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1
    h = h / np.sum(h)
    im1_marg = np.sum(h, axis=0)
    im2_marg = np.sum(h, axis=1)
    H_x = 0
    H_y = 0
    for i in range(N):
        if (im1_marg[i] != 0):
            H_x = H_x + im1_marg[i] * math.log2(im1_marg[i])
    for i in range(N):
        if (im2_marg[i] != 0):
            H_x = H_x + im2_marg[i] * math.log2(im2_marg[i])
    H_xy = 0
    for i in range(N):
        for j in range(N):
            if (h[i, j] != 0):
                H_xy = H_xy + h[i, j] * math.log2(h[i, j])
    MI = H_xy - H_x - H_y
    return MI