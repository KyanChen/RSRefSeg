from typing import Optional, Sequence, Dict
import torch
from mmengine import mkdir_or_exist, MMLogger, print_log
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from prettytable import PrettyTable
from mmseg.registry import METRICS


@METRICS.register_module()
class RefSegIoUMetric(BaseMetric):
    default_prefix = 'RefSeg'
    def __init__(self,
                 ignore_index: int = 255,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 prefix: Optional[str] = None
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label)
            self.results.append(self.intersect_and_union(pred_label, label, num_classes, self.ignore_index))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect, total_area_union, total_area_pred_label, total_area_label = results

        return_metrics = dict()
        # calculate the cumulative IoU and the generalized IoU
        cIoU = sum(total_area_intersect) / sum(total_area_union)
        per_image_iou = [area_intersect / area_union for area_intersect, area_union in zip(total_area_intersect, total_area_union)]
        gIoU = sum(per_image_iou) / len(per_image_iou)

        for id_cls in range(len(cIoU)):
            return_metrics[f'cIoU_{id_cls}'] = cIoU[id_cls].item()*100
            return_metrics[f'gIoU_{id_cls}'] = gIoU[id_cls].item()*100


        # calculate the segmentation accuracy for each IoU threshold (0.5, 0.6, 0.7, 0.8, 0.9)
        seg_iou_list = [0.5, 0.6, 0.7, 0.8, 0.9]
        per_image_iou = torch.stack(per_image_iou)  # BxNC
        for i, iou in enumerate(seg_iou_list):
            seg_correct = (per_image_iou > iou).sum(dim=0).float() / len(per_image_iou)
            for id_cls, correct in enumerate(seg_correct):
                return_metrics[f'seg_acc_{iou}_{id_cls}'] = correct.item()*100

        class_table_data = PrettyTable()
        class_table_data.field_names = ['class', 'cIoU', 'gIoU'] + [f'seg_acc_{iou}' for iou in seg_iou_list]
        for id_cls, class_name in enumerate(self.dataset_meta['classes']):
            class_table_data.add_row([class_name, round(return_metrics[f'cIoU_{id_cls}'], 2), round(return_metrics[f'gIoU_{id_cls}'], 2)] + [round(return_metrics[f'seg_acc_{iou}_{id_cls}'], 2) for iou in seg_iou_list])


        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return return_metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor, num_classes: int, ignore_index: int):
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_label = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label