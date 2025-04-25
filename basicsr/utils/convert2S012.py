import cv2
import torch
import random
import torchvision.transforms as transforms
import numpy as np
import polanalyser as pa
def I2S012(img_list,save_img=False,use_loss=False,clip=True,improve_contrast=True,dim_type="NCHW"):
    if isinstance(img_list,list):
        img_0,img_1,img_2,img_3=img_list
    else:
        if dim_type=="NCHW":
            img_0=img_list[:,0:3,:,:]
            img_1=img_list[:,3:6,:,:]
            img_2=img_list[:,6:9,:,:]
            img_3=img_list[:,9:12,:,:]
        elif dim_type=="NHWC":
            img_0=img_list[:,:,:,0:3]
            img_1=img_list[:,:,:,3:6]
            img_2=img_list[:,:,:,6:9]
            img_3=img_list[:,:,:,9:12]
        if dim_type=="CHW":
            img_0=img_list[0:3,:,:]
            img_1=img_list[3:6,:,:]
            img_2=img_list[6:9,:,:]
            img_3=img_list[9:12,:,:]
        elif dim_type=="HWC":
            img_0=img_list[:,:,0:3]
            img_1=img_list[:,:,3:6]
            img_2=img_list[:,:,6:9]
            img_3=img_list[:,:,9:12]
        elif dim_type=="N3HW":
            img_list=chunkbatch2dim(img_list)
            img_0=img_list[:,0:3,:,:]
            img_1=img_list[:,3:6,:,:]
            img_2=img_list[:,6:9,:,:]
            img_3=img_list[:,9:12,:,:]
    S0=0.5*(img_0+img_1+img_2+img_3)
    S1=img_0-img_2
    S2=img_1-img_3
    DOLP=torch.sqrt(S1**2+S2**2)/(S0+1e-5)
    if improve_contrast:
        DOLP=contrast(DOLP)
    if clip:
        S0=torch.clip(S0,0,1)
        DOLP=torch.clip(DOLP,0,1)
    if use_loss:
        # S0=0.5*S0
        DOLP=torch.sqrt(S1**2+S2**2+1e-5)
        # DOLP=torch.clip(DOLP,0,1)
        # S0=torch.clip(S0,0,1)
        # DOLP=torch.sqrt(S1**2+S2**2+1e-10)
    if save_img:
        return S0,DOLP
    return S0,S1,S2
def Gen_CPFA(full_img):
    batch,_,height, width=full_img.shape
    polar_order = [2, 1, 3, 0]
    color_order = [0, 1, 1, 2]
    mask = torch.zeros(batch,4, 3, height, width).to(full_img.device)
    # top left (tl)
    mask[:,polar_order[0], color_order[0],  ::4,  ::4] = 1 #polar 90 R
    mask[:,polar_order[1], color_order[0],  ::4, 1::4] = 1 #polar 45 R
    mask[:,polar_order[2], color_order[0], 1::4,  ::4] = 1 #polar 135 R
    mask[:,polar_order[3], color_order[0], 1::4, 1::4] = 1 #polar 0 R
    # top right (tr)
    mask[:,polar_order[0], color_order[1],  ::4, 2::4] = 1 #polar 90 G
    mask[:,polar_order[1], color_order[1],  ::4, 3::4] = 1 #polar 45 G
    mask[:,polar_order[2], color_order[1], 1::4, 2::4] = 1 #polar 135 G
    mask[:,polar_order[3], color_order[1], 1::4, 3::4] = 1 #polar 0 G
    # bottom left (bl)
    mask[:,polar_order[0], color_order[2], 2::4,  ::4] = 1 #polar 90 G
    mask[:,polar_order[1], color_order[2], 2::4, 1::4] = 1 #polar 45 G
    mask[:,polar_order[2], color_order[2], 3::4,  ::4] = 1 #polar 135 G
    mask[:,polar_order[3], color_order[2], 3::4, 1::4] = 1 #polar 0 G
    # bottom right (br)
    mask[:,polar_order[0], color_order[3], 2::4, 2::4] = 1 #polar 90 B
    mask[:,polar_order[1], color_order[3], 2::4, 3::4] = 1 #polar 45 B
    mask[:,polar_order[2], color_order[3], 3::4, 2::4] = 1 #polar 135 B
    mask[:,polar_order[3], color_order[3], 3::4, 3::4] = 1 #polar 0 B
    full_img_4_3=torch.concat([full_img[:,0:3,:,:].unsqueeze(dim=1),full_img[:,3:6,:,:].unsqueeze(dim=1),full_img[:,6:9,:,:].unsqueeze(dim=1),full_img[:,9:12,:,:].unsqueeze(dim=1)],dim=1)
    cpfa= torch.unsqueeze(torch.sum(full_img_4_3*mask,(1, 2)), 1)
    return cpfa
def CPFA_downsample(CPFA,color_type='rggb',polarization_type='[90,45;135,0]',output_dim=4):
    assert color_type == 'rggb', ('only support rggb!')
    assert polarization_type=='[90,45;135,0]', ('only support [90,45;135,0]!')
    Bayer_90=CPFA[:,:,0::2,0::2]
    Bayer_45=CPFA[:,:,0::2,1::2]
    Bayer_135=CPFA[:,:,1::2,0::2]
    Bayer_0 = CPFA[:,:,1::2,1::2]
    Bayer_90_init=get_initrgb(Bayer_90,output_dim)
    Bayer_45_init=get_initrgb(Bayer_45,output_dim)
    Bayer_135_init=get_initrgb(Bayer_135,output_dim)
    Bayer_0_init=get_initrgb(Bayer_0,output_dim)
    result=torch.concat([Bayer_0_init,Bayer_45_init,Bayer_90_init,Bayer_135_init],dim=1)
    return result
def init_CPDM(cpfa):
    tfs=transforms.ToTensor()
    device=cpfa.device
    img_full_list = []
    for i in range(cpfa.shape[0]):
        single_cpfa = cpfa[i]
        single_cpfa = single_cpfa.squeeze(0)
        single_cpfa = single_cpfa * 255
        single_cpfa = single_cpfa.detach().cpu().numpy()
        single_cpfa = np.array(single_cpfa, dtype="uint8")
        img_000, img_045, img_090, img_135 = pa.demosaicing(single_cpfa, pa.COLOR_PolarRGB)
        img_000 = cv2.cvtColor(img_000, cv2.COLOR_BGR2RGB)
        img_045 = cv2.cvtColor(img_045, cv2.COLOR_BGR2RGB)
        img_090 = cv2.cvtColor(img_090, cv2.COLOR_BGR2RGB)
        img_135 = cv2.cvtColor(img_135, cv2.COLOR_BGR2RGB)
        img_000 = tfs(img_000)
        img_045 = tfs(img_045)
        img_090 = tfs(img_090)
        img_135 = tfs(img_135)
        img_000 = img_000.unsqueeze(0)
        img_045 = img_045.unsqueeze(0)
        img_090 = img_090.unsqueeze(0)
        img_135 = img_135.unsqueeze(0)
        img_full = torch.concat([img_000, img_045, img_090, img_135], dim=1).to(device)
        img_full_list.append(img_full)
    img_full_batch = torch.concat(img_full_list, dim=0)
    return img_full_batch
def contrast(img,a=0.3):
    img_clone = img.clone()
    img_clone[img>a] = a
    img_clone[img<0] = 0
    img_clone = img_clone/a
    return img_clone
def get_initrgb(img):
    r=img[:,:,0::2,0::2]
    g_0=img[:,:,0::2,1::2]
    g_1=img[:,:,1::2,0::2]
    b=img[:,:,1::2,1::2]
    g=(g_0+g_1)/2
    init_img=torch.concat([r,g,b],dim=1)
    return init_img
def get_ycbcr(img_list):
    if isinstance(img_list,list):
        img_0,img_1,img_2,img_3=img_list
    else:
        img_0=img_list[:,0:3,:,:]
        img_1=img_list[:,3:6,:,:]
        img_2=img_list[:,6:9,:,:]
        img_3=img_list[:,9:12,:,:]
    img_0=rgb_to_ycbcr(img_0)
    img_1=rgb_to_ycbcr(img_1)
    img_2=rgb_to_ycbcr(img_2)
    img_3=rgb_to_ycbcr(img_3)
    return torch.concat([img_0,img_1,img_2,img_3],dim=1)
def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

>> Meta_results = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)
def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)
def chunkdim2batch(img,chunk_size=4):
    chunks = torch.chunk(img, chunk_size, dim=1)
    result = torch.cat(chunks, dim=0)
    return result
def chunkbatch2dim(img,chunk_size=4):
    chunks = torch.chunk(img, chunk_size, dim=0)
    result = torch.cat(chunks, dim=1)
    return result