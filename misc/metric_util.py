
import numpy as np
from mmengine import MMLogger
logger = MMLogger.get_instance('selfocc')
import torch.distributed as dist
import torch
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

class MeanIoU:

    def __init__(self,
                 class_indices,
                #  ignore_label: int,
                 empty_label,
                 label_str,
                 use_mask=False,
                 dataset_empty_label=17,
                 filter_minmax=True,
                 name = 'none'):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        # self.ignore_label = ignore_label
        self.empty_label = empty_label
        self.dataset_empty_label = dataset_empty_label
        self.label_str = label_str
        self.use_mask = use_mask
        self.filter_minmax = filter_minmax
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes+1).cuda()
        self.total_correct = torch.zeros(self.num_classes+1).cuda()
        self.total_positive = torch.zeros(self.num_classes+1).cuda()
        
    def _after_step(self, outputs, targets, mask=None):
        # outputs = outputs[targets != self.ignore_label]
        # targets = targets[targets != self.ignore_label]
        if not isinstance(targets, (torch.Tensor, np.ndarray)):
            assert mask is None
            labels = torch.from_numpy(targets['semantics']).cuda()
            masks = torch.from_numpy(targets['mask_camera']).bool().cuda()
            targets = labels
            targets[targets == self.dataset_empty_label] = self.empty_label
            if self.filter_minmax:
                max_z = (targets != self.empty_label).nonzero()[:, 2].max()
                min_z = (targets != self.empty_label).nonzero()[:, 2].min()
                outputs[..., (max_z + 1):] = self.empty_label
                outputs[..., :min_z] = self.empty_label
            if self.use_mask:
                outputs = outputs[masks]
                targets = targets[masks]
        else:
            if mask is not None:
                outputs = outputs[mask]
                targets = targets[mask]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()
        
        self.total_seen[-1] += torch.sum(targets != self.empty_label).item()
        self.total_correct[-1] += torch.sum((targets != self.empty_label)
                                            & (outputs != self.empty_label)).item()
        self.total_positive[-1] += torch.sum(outputs != self.empty_label).item()

    def _after_epoch(self):
        if dist.is_initialized():
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
            dist.barrier()

        
        total_seen = self.total_seen.cpu().numpy()
        total_correct = self.total_correct.cpu().numpy()
        total_positive = self.total_positive.cpu().numpy()

        ious = []
        ret_dict = dict()

        header = ['classes']
        for i in range(len(self.label_str)):
            header.append(self.label_str[i])
        header.extend(['miou', 'iou'])
        table_columns = [['results']]

        for i in range(self.num_classes): # todo 只计算语义类，不包括非空类
            if self.total_seen[i] == 0: # todo iou & recall
                cur_iou = np.nan
            else:
                cur_iou = total_correct[i] / (total_seen[i] + total_positive[i] - total_correct[i]) # todo iou = TP / (TP + FN + FP)

            ious.append(cur_iou)
            table_columns.append([f'{cur_iou:.4f}'])
            
            ret_dict[self.label_str[i]] = cur_iou * 100

        miou = np.nanmean(ious)
        iou = total_correct[-1] / (total_seen[-1] + total_positive[-1] - total_correct[-1])

        table_columns.append([f'{miou:.4f}'])
        table_columns.append([f"{iou:.4f}"])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table)        
        
        ret_dict['miou'] = miou * 100
        ret_dict['iou'] = iou * 100
        
        # return miou * 100, iou * 100
        return ret_dict
