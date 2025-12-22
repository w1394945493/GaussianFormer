from mmseg.models import SEGMENTORS, builder
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule


@SEGMENTORS.register_module()
class CustomBaseSegmentor(BaseModule):

    def __init__(
        self,
        img_backbone=None,
        img_neck=None,
        lifter=None,
        encoder=None,
        head=None, 
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if img_backbone is not None:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            try:
                self.img_neck = builder.build_neck(img_neck)
            except:
                self.img_neck = MODELS.build(img_neck)
        # todo -------------------------------------#
        if lifter is not None:
            # todo lifter: 包括了 论文中 关于 高斯属性与查询向量相关的部分
            self.lifter = builder.build_head(lifter) # todo GaussianLifter
        # todo -------------------------------------#
        # todo 主要的设计部分：自编码模块、图像交叉注意力模块和细化模块 3.2节部分内容在这里
        if encoder is not None: # todo 编码层：结构为# todo ['deformable', 'ffn', 'norm', 'refine', 'spconv', 'norm',| 'deformable', 'ffn', 'norm', 'refine', 'spconv', 'norm',| 'deformable', 'ffn', 'norm', 'refine', 'spconv', 'norm',| 'deformable', 'ffn', 'norm', 'refine']
            self.encoder = builder.build_head(encoder)
        # todo -------------------------------------#
        # todo 3.3节 高斯到体素投影模块设计
        if head is not None: # todo head头，主要包括 高斯 -> 体素 操作，损失部分
            self.head = builder.build_head(head)

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats = self.img_backbone(imgs)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return {'ms_img_feats': img_feats_reshaped}

    def forward(
        self,
        imgs,
        metas,
        **kwargs
    ):
        pass