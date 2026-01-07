# Copyright (c) OpenMMLab. All rights reserved.
# 设置进程名
from setproctitle import setproctitle
setproctitle("wys")

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['RANK'] = '0'
import argparse



import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

import pickle

# from vis import save_occ
def pass_print(*args, **kwargs):
    pass


def save_occ(pred_occ,gt_occ,file_path):
    output = dict(
        occ_pred = pred_occ.cpu().numpy(),
        occ_gt = gt_occ.cpu().numpy(),
    )
    # torch.save(output, file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(output, f)







def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    os.makedirs(args.work_dir,exist_ok=True)

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1

    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        if torch.cuda.is_available():
            my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)

    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        raw_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
        print(f'successfully resumed.')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        try:
            print(raw_model.load_state_dict(state_dict, strict=False))
        except:
            from misc.checkpoint_util import refine_load_from_sd
            print(raw_model.load_state_dict(
                refine_load_from_sd(state_dict), strict=False))

    print_freq = cfg.print_freq
    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17, #17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
         True, 17, filter_minmax=False)
    
    miou_metric.reset()

    my_model.eval()
    os.environ['eval'] = 'true'

    with torch.no_grad():
        for i_iter_val, data in tqdm(enumerate(val_dataset_loader),total=len(val_dataset_loader),desc="Processing"):

            # todo ----------------------------------#
            if torch.cuda.is_available():
                for k in list(data.keys()):
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].cuda()
            
            input_imgs = data.pop('img')
            # todo ----------------------------------#
            result_dict = my_model(imgs=input_imgs, metas=data) # todo input_imgs: (b v 3 h w) h=864 w=1600
            if 'final_occ' in result_dict:
                for idx, pred in enumerate(result_dict['final_occ']):
                    pred_occ = pred # todo (640000)
                    gt_occ = result_dict['sampled_label'][idx]
                    occ_mask = result_dict['occ_mask'][idx].flatten()

                    # todo --------------------------#
                    # todo 可视化
                    if args.vis_occ:
                        save_dir = os.path.join(args.work_dir, 'vis')
                        os.makedirs(save_dir, exist_ok=True)

                        scene_token,token = data['scene_token'][idx],data['token'][idx]
                        file_name = f'{scene_token}_{token}'
                        # scene_token,sample_idx,token = data['scene_token'][idx],data['sample_idx'][idx],data['token'][idx]
                        # file_name = f'{scene_token}_{sample_idx}_{token}'
                        #  file_path = os.path.join(save_dir,f'{file_name}.pth')
                        file_path = os.path.join(save_dir,f'{file_name}.pkl')
                        save_occ(
                            pred_occ = pred_occ.reshape(200, 200, 16),
                            gt_occ = gt_occ.reshape(200, 200, 16),
                            file_path = file_path
                        )

                    #     save_occ(
                    #         os.path.join(args.work_dir, 'vis'),
                    #         pred_occ.reshape(1, 200, 200, 16),
                    #         f'val_{i_iter_val}_pred',
                    #         True, 0)
                    #     save_occ(
                    #         os.path.join(args.work_dir, 'vis'),
                    #         gt_occ.reshape(1, 200, 200, 16),
                    #         f'val_{i_iter_val}_gt',
                    #         True, 0)
                    # todo ------------------------#
                    # todo 评估
                    miou_metric._after_step(pred_occ, gt_occ, occ_mask)
                    # breakpoint()

            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))

    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    miou_metric.reset()

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
