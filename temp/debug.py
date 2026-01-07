import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mmengine import Config
from mmseg.models import build_segmentor

import model
from dataset import get_dataloader
from loss import OPENOCC_LOSS

py_config = '/home/lianghao/wangyushen/Projects/GaussianFormer/config/nuscenes_gs25600_solid_custom.py'
cfg = Config.fromfile(py_config)
model = build_segmentor(cfg.model)

ckpt_path = '/home/lianghao/wangyushen/data/wangyushen/Output/debug/my_model.pth'
ckpt = torch.load(ckpt_path,map_location='cpu')

model.load_state_dict(ckpt)