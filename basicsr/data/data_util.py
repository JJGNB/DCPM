import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F

from basicsr.data.transforms import mod_crop
from basicsr.utils import img2tensor, scandir


def read_img_seq(path, require_mod_crop=False, scale=1, return_imgname=False):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Default False.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
        list[str]: Returned image name list.
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]

    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs


def generate_frame_indices(crt_idx, max_frame_num, num_frames, padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    ::

        lq.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(f'{input_key} folder and {gt_key} folder should both in lmdb '
                         f'formats. But received {input_key}: {input_folder}; '
                         f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(dict([(f'{input_key}_path', lmdb_key), (f'{gt_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.strip().split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, f'The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}'
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, f'{input_name} is not in {input_key}_paths.'
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths
def pairedIVIF_paths_from_lmdb(folders, keys):
        """Generate paired paths from lmdb files.

        Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

        ::

            lq.lmdb
            ├── data.mdb
            ├── lock.mdb
            ├── meta_info.txt

        The data.mdb and lock.mdb are standard lmdb files and you can refer to
        https://lmdb.readthedocs.io/en/release/ for more details.

        The meta_info.txt is a specified txt file to record the meta information
        of our datasets. It will be automatically created when preparing
        datasets by our provided dataset tools.
        Each line in the txt file records
        1)image name (with extension),
        2)image shape,
        3)compression level, separated by a white space.
        Example: `baboon.png (120,125,3) 1`

        We use the image name without extension as the lmdb key.
        Note that we use the same key for the corresponding lq and gt images.

        Args:
            folders (list[str]): A list of folder path. The order of list should
                be [input_folder, gt_folder].
            keys (list[str]): A list of keys identifying folders. The order should
                be in consistent with folders, e.g., ['lq', 'gt'].
                Note that this key is different from lmdb keys.

        Returns:
            list[str]: Returned path list.
        """
        assert len(folders) == 3, ('The len of folders should be 3 with [ir_folder, vi_folder, ve_folder]. '
                                f'But got {len(folders)}')
        assert len(keys) == 3, f'The len of keys should be 2 with [ir_key, vi_key, ve_key]. But got {len(keys)}'
        ir_folder, vi_folder,ve_folder = folders
        ir_key, vi_key, ve_key  = keys

        if not (ir_folder.endswith('.lmdb') and ve_folder.endswith('.lmdb')):
            raise ValueError(f'{ir_key} folder and {ve_key} folder should both in lmdb '
                            f'formats. But received {ir_key}: {ir_folder}; '
                            f'{ve_key}: {ve_folder}')
        # ensure that the two meta_info files are the same
        with open(osp.join(ir_folder, 'meta_info.txt')) as fin:
            ir_lmdb_keys = [line.split('.')[0] for line in fin]
        with open(osp.join(vi_folder, 'meta_info.txt')) as fin:
            vi_lmdb_keys = [line.split('.')[0] for line in fin]
        with open(osp.join(ve_folder, 'meta_info.txt')) as fin:
            ve_lmdb_keys = [line.split('.')[0] for line in fin]
        if set(vi_lmdb_keys) != set(ve_lmdb_keys):
            raise ValueError(f'Keys in {vi_key}_folder and {ve_key}_folder are different.')
        else:
            paths = []
            for lmdb_key in sorted(vi_lmdb_keys):
                paths.append(dict([(f'{ir_key}_path', lmdb_key),(f'{vi_key}_path', lmdb_key), (f'{ve_key}_path', lmdb_key)]))
            return paths
def pairedCPDM_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 9, ('The len of folders should be 8 with [self.gt_folder_0,self.gt_folder_45,self.gt_folder_90,self.gt_folder_135,self.lq_folder_0,self.lq_folder_45,self.lq_folder_90,self.lq_folder_135,self.lq_folder_cpfa]. '
                               f'But got {len(folders)}')
    assert len(keys) == 9, f'The len of keys should be 3 with [gt_0, gt_45, gt_90,gt_135,lq_0,lq_45,lq_90,lq_135,lq_fpfa]. But got {len(keys)}'
    gt_folder_0,gt_folder_45,gt_folder_90,gt_folder_135,lq_folder_0,lq_folder_45,lq_folder_90,lq_folder_135,lq_folder_cpfa= folders
    gt_0_key, gt_45_key, gt_90_key,gt_135_key,lq_0_key,lq_45_key,lq_90_key,lq_135_key,lq_cpfa_key= keys

    gt_paths_0 = list(scandir(gt_folder_0))
    gt_paths_45 = list(scandir(gt_folder_45))
    gt_paths_90 = list(scandir(gt_folder_90))
    gt_paths_135 = list(scandir(gt_folder_135))
    paths = []
    for gt_path_0,gt_path_45,gt_path_90,gt_path_135 in zip(gt_paths_0,gt_paths_45,gt_paths_90,gt_paths_135):
        basename_0, ext_0 = osp.splitext(osp.basename(gt_path_0))
        basename_45, ext_45 = osp.splitext(osp.basename(gt_path_45))
        basename_90, ext_90 = osp.splitext(osp.basename(gt_path_90))
        basename_135, ext_135 = osp.splitext(osp.basename(gt_path_135))
        gt_name_0 = f'{filename_tmpl.format(basename_0)}{ext_0}'
        gt_name_45 = f'{filename_tmpl.format(basename_45)}{ext_45}'
        gt_name_90 = f'{filename_tmpl.format(basename_90)}{ext_90}'
        gt_name_135 = f'{filename_tmpl.format(basename_135)}{ext_135}'
        g_path_0 = osp.join(gt_folder_0, gt_name_0)
        g_path_45 = osp.join(gt_folder_45, gt_name_45)
        g_path_90 = osp.join(gt_folder_90, gt_name_90)
        g_path_135 = osp.join(gt_folder_135, gt_name_135)
        l_path_0 = osp.join(lq_folder_0, gt_name_0)
        l_path_45 = osp.join(lq_folder_45, gt_name_45)
        l_path_90 = osp.join(lq_folder_90, gt_name_90)
        l_path_135 = osp.join(lq_folder_135, gt_name_135)
        l_path_cpfa = osp.join(lq_folder_cpfa, gt_name_135)
        # l_path_cpfa = osp.join(lq_folder_cpfa, gt_name_135.replace("_135.png","_CPFA.png"))
        paths.append(dict([(f'{gt_0_key}_path', g_path_0), (f'{gt_45_key}_path', g_path_45), (f'{gt_90_key}_path', g_path_90),(f'{gt_135_key}_path', g_path_135),(f'{lq_0_key}_path', l_path_0),(f'{lq_45_key}_path', l_path_45),(f'{lq_90_key}_path', l_path_90),(f'{lq_135_key}_path', l_path_135),(f'{lq_cpfa_key}_path', l_path_cpfa)]))
    return paths
def pairedIVIF_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 3, ('The len of folders should be 3 with [ir_folder, vi_folder, ve_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 3, f'The len of keys should be 3 with [ir_key, vi_key, ve_key]. But got {len(keys)}'
    ir_folder, vi_folder, ve_folder = folders
    ir_key, vi_key, ve_key = keys

    ir_paths = list(scandir(ir_folder))
    vi_paths = list(scandir(vi_folder))
    ve_paths = list(scandir(ve_folder))
    assert len(ir_paths) == len(vi_paths), (f'{ir_key} and {vi_key} datasets have different number of images: '
                                               f'{len(ir_paths)}, {len(vi_paths)}.')
    paths = []
    for ve_path in ve_paths:
        basename, ext = osp.splitext(osp.basename(ve_path))
        ir_name = f'{filename_tmpl.format(basename)}{ext}'
        ir_path = osp.join(ir_folder, ir_name)
        vi_name = f'{filename_tmpl.format(basename)}{ext}'
        vi_path = osp.join(vi_folder, vi_name)
        assert ir_name in ir_paths, f'{ir_name} is not in {ir_key}_paths.'
        assert vi_name in vi_paths, f'{vi_name} is not in {vi_key}_paths.'
        ve_path = osp.join(ve_folder, vi_name)
        paths.append(dict([(f'{ir_key}_path', ir_path), (f'{vi_key}_path', vi_path), (f'{ve_key}_path', ve_path)]))
    return paths


def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3, 4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x
