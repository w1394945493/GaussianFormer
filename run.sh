# todo occ标注生成(参考自SurroundOcc)
python /home/lianghao/wangyushen/Projects/GaussianFormer/tools/generate_occupancy_nuscenes/generate_mini_occ_nuscenes.py \
    --config_path /home/lianghao/wangyushen/Projects/GaussianFormer/tools/generate_occupancy_nuscenes/config.yaml \
    --label_mapping /home/lianghao/wangyushen/Projects/GaussianFormer/tools/generate_occupancy_nuscenes/nuscences.yaml \
    --version v1.0-mini \
    --dataroot /home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini \
    --save_path /home/lianghao/wangyushen/data/wangyushen/Datasets/data/gt_occ_v1.0-mini \



# todo GaussianFormer 评估 示例
python eval.py --py-config config/xxxx.py --work-dir out/xxxx/ --resume-from out/xxxx/state_dict.pth

python /home/lianghao/wangyushen/Projects/GaussianFormer/eval.py \
    --py-config /home/lianghao/wangyushen/Projects/GaussianFormer/config/nuscenes_gs25600_solid_custom.py \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gaussianformer/debug \
    --resume-from /home/lianghao/wangyushen/data/wangyushen/Weights/gaussianformer/nuscenes_gs25600_solid/state_dict.pth \
    --vis-occ

# todo GaussianFormer 训练 示例
python train.py --py-config config/xxxx.py --work-dir out/xxxx

python /home/lianghao/wangyushen/Projects/GaussianFormer/train.py \
    --py-config /home/lianghao/wangyushen/Projects/GaussianFormer/config/nuscenes_gs25600_solid_custom.py \
    --work-dir /home/lianghao/wangyushen/data/wangyushen/Output/gaussianformer/debug \