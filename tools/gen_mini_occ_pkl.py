from copy import deepcopy
import mmengine
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

# todo 用于生成GaussFormer使用的pkl文件
if __name__=='__main__':

    ori_mini_val_pkl_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini/nuscenes_mini_infos_val.pkl'
    ori_mini_val_data = mmengine.load(ori_mini_val_pkl_path)
    val_scene_set = set()  # 用集合收集不同场景，自动去重

    for item in ori_mini_val_data['data_list']:
        scene_token = item.get('scene_token')  # 取出场景标识
        if scene_token is not None:
            val_scene_set.add(scene_token)

    print(f"v1.0min 共有 {len(val_scene_set)} 个不同的val场景，分别是：")
    for scene in val_scene_set:
        print(scene)

    ori_mini_train_pkl_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini/nuscenes_mini_infos_train.pkl'
    ori_mini_train_data = mmengine.load(ori_mini_train_pkl_path)
    train_scene_set = set()  # 用集合收集不同场景，自动去重

    for item in ori_mini_train_data['data_list']:
        scene_token = item.get('scene_token')  # 取出场景标识
        if scene_token is not None:
            train_scene_set.add(scene_token)

    print(f"v1.0min 共有 {len(train_scene_set)} 个不同的train场景，分别是：")
    for scene in train_scene_set:
        print(scene)

    # ========================== 生成 mini val ==========================
    val_pkl_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/nuscenes/nuscenes_infos_val_sweeps_occ.pkl'
    save_mini_val_pkl_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/mini/nuscenes_mini_infos_val_sweeps_occ.pkl'

    val_data = mmengine.load(val_pkl_path)

    train_pkl_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/nuscenes/nuscenes_infos_train_sweeps_occ.pkl'
    save_mini_train_pkl_path = '/home/lianghao/wangyushen/data/wangyushen/Datasets/data/nuscenes_cam/mini/nuscenes_mini_infos_train_sweeps_occ.pkl'

    train_data = mmengine.load(train_pkl_path)

    data = val_data.copy()
    data['infos'].update(train_data['infos'])
    data['metadata'].extend(train_data['metadata'])

    val_scene_infos = data['infos']
    val_keyframes = data['metadata']
    val_keyframes = sorted(val_keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))

    mini_nusc = NuScenes(
        version='v1.0-mini',
        dataroot='/home/lianghao/wangyushen/data/wangyushen/Datasets/data/v1.0-mini',
        verbose=False
    )
    mini_scenes_infos = [scene['token'] for scene in mini_nusc.scene]
    print(f"v1.0min 共有 {len(mini_scenes_infos)} 个不同的train+val场景，分别是：")
    for scene in mini_scenes_infos:
        print(scene)

    mini_val_pkl = {'infos': {}, 'metadata': []}

    print('val:')
    for val_scene_info, val_scene_data in tqdm(val_scene_infos.items()):
        # if val_scene_info in mini_scenes_infos:
        if val_scene_info in val_scene_set:
            print(val_scene_info)
            mini_val_pkl['infos'][val_scene_info] = val_scene_data
            matched_keyframes = [kf for kf in val_keyframes if kf[0] == val_scene_info]
            mini_val_pkl['metadata'].extend(matched_keyframes)

    mmengine.dump(mini_val_pkl, save_mini_val_pkl_path)
    print(f"✅ 已保存 mini val 到: {save_mini_val_pkl_path}")


    # ========================== 生成 mini train ==========================


    train_scene_infos = data['infos']
    train_keyframes = data['metadata']
    train_keyframes = sorted(train_keyframes, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))

    mini_train_pkl = {'infos': {}, 'metadata': []}

    print('train:')
    for train_scene_info, train_scene_data in tqdm(train_scene_infos.items()):
        # if train_scene_info in mini_scenes_infos:
        if train_scene_info in train_scene_set:
            print(train_scene_info)
            mini_train_pkl['infos'][train_scene_info] = train_scene_data
            matched_keyframes = [kf for kf in train_keyframes if kf[0] == train_scene_info]
            mini_train_pkl['metadata'].extend(matched_keyframes)

    mmengine.dump(mini_train_pkl, save_mini_train_pkl_path)
    print(f"✅ 已保存 mini train 到: {save_mini_train_pkl_path}")
