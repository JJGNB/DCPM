from torch.utils import data as data
from os import path as osp
from basicsr.data.data_util import pairedCPDM_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
import torch.nn.functional as F
from basicsr.utils.registry import DATASET_REGISTRY
import torch

@DATASET_REGISTRY.register(suffix='basicsr')
class CPDMImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(CPDMImageDataset, self).__init__()
        self.opt = opt
        self.paths=[]
        if opt.get('data_source', None) is not None:
            for ii in range(len(opt['data_source'])):
                configs = opt['data_source'].get(f'source{ii+1}')
                # file client (io backend)
                self.file_client = None
                self.io_backend_opt = configs['io_backend']
                self.mean = configs['mean'] if 'mean' in configs else None
                self.std = configs['std'] if 'std' in configs else None
                self.use_lq=configs['use_lq']
                self.use_cpfa=configs['use_cpfa']
                self.gt_folder_0=configs['dataroot_gt_0']
                self.gt_folder_45=configs['dataroot_gt_45']
                self.gt_folder_90=configs['dataroot_gt_90']
                self.gt_folder_135=configs['dataroot_gt_135']
                self.lq_folder_0 = configs['dataroot_lq_0']
                self.lq_folder_45 = configs['dataroot_lq_45']
                self.lq_folder_90 = configs['dataroot_lq_90']
                self.lq_folder_135 = configs['dataroot_lq_135']
                self.lq_folder_cpfa = configs['dataroot_lq_cpfa']
                if 'filename_tmpl' in configs:
                    self.filename_tmpl = configs['filename_tmpl']
                else:
                    self.filename_tmpl = '{}'
                path_temp=pairedCPDM_paths_from_folder([self.gt_folder_0,self.gt_folder_45,self.gt_folder_90,self.gt_folder_135,self.lq_folder_0,self.lq_folder_45,self.lq_folder_90,self.lq_folder_135,self.lq_folder_cpfa], ['gt_0', 'gt_45', 'gt_90','gt_135','lq_0','lq_45','lq_90','lq_135','lq_cpfa'], self.filename_tmpl)
                self.paths=self.paths+path_temp
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_0_path = self.paths[index]['gt_0_path']
        img_bytes = self.file_client.get(gt_0_path, 'gt_0')
        img_gt_0 = imfrombytes(img_bytes, float32=True)
        gt_45_path = self.paths[index]['gt_45_path']
        img_bytes = self.file_client.get(gt_45_path, 'gt_45')
        img_gt_45 = imfrombytes(img_bytes, float32=True)
        gt_90_path = self.paths[index]['gt_90_path']
        img_bytes = self.file_client.get(gt_90_path, 'gt_90')
        img_gt_90 = imfrombytes(img_bytes, float32=True)
        gt_135_path = self.paths[index]['gt_135_path']
        img_bytes = self.file_client.get(gt_135_path, 'gt_135')
        img_gt_135 = imfrombytes(img_bytes, float32=True)
        lq_0_path = self.paths[index]['lq_0_path']
        img_bytes = self.file_client.get(lq_0_path, 'lq_0')
        img_lq_0 = imfrombytes(img_bytes, float32=True)
        lq_45_path = self.paths[index]['lq_45_path']
        img_bytes = self.file_client.get(lq_45_path, 'lq_45')
        img_lq_45 = imfrombytes(img_bytes, float32=True)
        lq_90_path = self.paths[index]['lq_90_path']
        img_bytes = self.file_client.get(lq_90_path, 'lq_90')
        img_lq_90 = imfrombytes(img_bytes, float32=True)
        lq_135_path = self.paths[index]['lq_135_path']
        img_bytes = self.file_client.get(lq_135_path, 'lq_135')
        img_lq_135 = imfrombytes(img_bytes, float32=True)
        lq_cpfa_path = self.paths[index]['lq_cpfa_path']
        img_bytes = self.file_client.get(lq_cpfa_path, 'lq_cpfa')
        img_lq_cpfa = imfrombytes(img_bytes, float32=True)
        img_gt_0= img2tensor(img_gt_0, bgr2rgb=True, float32=True,yonly=False)
        img_gt_45= img2tensor(img_gt_45, bgr2rgb=True, float32=True,yonly=False)
        img_gt_90 = img2tensor(img_gt_90, bgr2rgb=True, float32=True,yonly=False)
        img_gt_135= img2tensor(img_gt_135, bgr2rgb=True, float32=True,yonly=False)
        img_lq_0= img2tensor(img_lq_0, bgr2rgb=True, float32=True,yonly=False)
        img_lq_45= img2tensor(img_lq_45, bgr2rgb=True, float32=True,yonly=False)
        img_lq_90 = img2tensor(img_lq_90, bgr2rgb=True, float32=True,yonly=False)
        img_lq_135= img2tensor(img_lq_135, bgr2rgb=True, float32=True,yonly=False)
        img_lq_cpfa= img2tensor(img_lq_cpfa, bgr2rgb=False, float32=True,yonly=False)
        img_lq=torch.concat([img_lq_0,img_lq_45,img_lq_90,img_lq_135],dim=0)
        img_gt=torch.concat([img_gt_0,img_gt_45,img_gt_90,img_gt_135],dim=0)
        return {'gt': img_gt, 'lq': img_lq,'cpfa': img_lq_cpfa, 'lq_path': lq_0_path, 'gt_path': gt_0_path,'cpfa_path': lq_cpfa_path, 'txt': ""}

    def __len__(self):
        return len(self.paths)
