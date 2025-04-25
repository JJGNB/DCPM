# DCPM: Demosaicking customized diffusion model for snapshot polarization imaging

:star: If you've found DCPM useful for your research or projects, please show your support by starring this repo. Thanks! :hugs:

---
>For snapshot polarization imaging, the color polarization demosaicking is essential to reconstruct full resolution
from a mosaic array, which is the latest unsolved issue. Due to the mosaic array missing a large number of
key pixels, existing one-step deep learning-based methods exhibit limited demosaicking performance. Hence,
we make the first attempt to address the color polarization demosaicking task through the diffusion model,
namely DCPM. Specifically, we extend the residual-based diffusion process to the task of color polarization
demosaicking and improve the network architecture to accommodate full-resolution polarization images.
Moreover, considering the polarization property of images, a customized loss function is proposed to assist in
the diffusion model training. Extensive experiments on both synthetic and real-world benchmarks demonstrate
the effectiveness of the proposed method.
><img src="./assets/framework.png" align="middle" width="800">
---

## Update
- **2025.04.23**: DCPM_v2 is released. Compared to the version in the paper, we use [polanalyser](https://github.com/elerac/polanalyser) to generate the initial interpolation results and train our network using four datasets ([EARI](http://www.ok.sc.e.titech.ac.jp/res/PolarDem/index.html), [Qiu's](https://repository.kaust.edu.sa/items/3653d5cd-a78b-40d7-899e-57f5f137ca85), [PIDSR](https://github.com/PRIS-CV/PIDSR) and Ours)

## TODO
- The dataset we propose will be released at an appropriate time.

## Requirements
```
conda create -n dcpm python=3.10
conda activate dcpm
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
cd DCPM
pip install -r requirements.txt
python setup.py develop
```
## Inference
Download the weight from this [link](https://pan.baidu.com/s/1x-MjSdaC7dNMNtBrLvkYCA?pwd=ndb9) and put it in the folder of "checkpoints".
```
python ./inference/inference_DCPM.py -opt "./options/test/CPDM/test_for_CPDM.yml"
```
The test is performed on simulated data by default. If you want to test on real mosaic images, set "real_cpdm" to "true" in the configuration file, comment out lines 15-23. and uncomment line 14. Note that the mosaic image must be arranged in the RGGB format, following the pattern [90,45;135,0].


## Training
Download the dataset from this [link](https://pan.baidu.com/s/1x-MjSdaC7dNMNtBrLvkYCA?pwd=ndb9) and replace the "datasets" in the project with them. If you already have the initial interpolation results, such as those obtained through PCDP interpolation, ensure they are stored in the same format as the GT. In the config file, set "use_lq" to true and update the path for lq accordingly.

Begin training:
```
python ./basicsr/train_CPDM.py -opt "./options/train/CPDM/train_for_CPDM.yml"
```
The training results are stored in the "experiments" folder.

## Acknowledgement
This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [ResShift](https://github.com/zsyOAOA/ResShift). Thanks for their awesome works.

## Contact
If you have any questions, please feel free to contact me via "244603040@csu.edu.cn" or open an issue.
## References
If you find this repository useful for your research, please cite the following work.
```
@article{li2025demosaicking,
  title={Demosaicking customized diffusion model for snapshot polarization imaging},
  author={Li, Chenggong and Luo, Yidong and Wu, Caiyun and Zhang, Junchao and Yang, Degui and Zhao, Dangjun},
  journal={Optics \& Laser Technology},
  volume={188},
  pages={112868},
  year={2025},
  publisher={Elsevier}
}
```