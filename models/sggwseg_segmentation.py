# Project repository: https://github.com/2022jiangjiazheng
"""SGGWSeg task definition: losses, metrics, and model hyperparameters."""

import cv2
import numpy as np
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses.dice import DiceLoss
from skimage.morphology import skeletonize

from models.sggwseg import SGGWSegBase


class SkeletonLoss(nn.Module):
    """Class-balanced binary cross-entropy for sparse skeleton targets."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """Compare skeleton logits and binary targets with shape ``BCHW`` or ``BHW``."""
        # Align prediction and target shapes to [B, H, W].
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
            
        target = target.float()

        # A sparse skeleton yields a larger foreground weight (beta).
        beta = 1 - torch.mean(target)
        
        # target=1 -> beta; target=0 -> 1-beta.
        weights = 1 - beta + (2 * beta - 1) * target

        # Numerically stable BCE-with-logits written in log-sum-exp form.
        pos = (pred >= 0).float()
        bce_loss = torch.log(1 + (pred - 2 * pred * pos).exp()) - pred * (target - pos)

        # Preserve the original weighted mean calculation.
        weighted_loss = bce_loss * weights
        return torch.sum(weighted_loss) / (pred.numel() + 1e-8)
    
def generate_skeleton_edge(y, width=1):
    """Generate skeleton targets from binary masks on the original device.

    Args:
        y: Label tensor with shape ``[B, H, W]``.
        width: Skeleton width in pixels; ``1`` keeps a one-pixel skeleton.
    """
    device = y.device
    y_np = y.detach().cpu().numpy().astype(np.uint8)

    batch_edges = []
    for mask in y_np:
        binary_mask = (mask > 0).astype(np.uint8)

        skeleton = skeletonize(binary_mask).astype(np.uint8)

        if width > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
            skeleton = cv2.dilate(skeleton, kernel)

        batch_edges.append(skeleton)

    edge_tensor = torch.from_numpy(np.array(batch_edges)).to(device).long()
    return edge_tensor


class SGGWSeg(SGGWSegBase):
    """Great Wall segmentation model used for training and inference."""

    def __init__(self, hparams):
        super().__init__(hparams=hparams, metric=None, in_channel=3, num_classes=2)

    def make_batch_dictionary(self, loss, metric, name_of_loss):
        """Build the per-step output consumed by epoch-level logging."""
        return {
            name_of_loss: loss,
            "IoU": metric[0],
            "IoU non-greatwall": metric[1],
            "IoU greatwall": metric[2],
        }

    def log_metric(self, outputs, train_or_val_or_test):
        """Average and log overall/background/Great-Wall IoU values."""
        avg_iou = torch.stack([x["IoU"] for x in outputs]).mean()
        avg_iou_non_greatwall = torch.stack([x["IoU non-greatwall"] for x in outputs]).mean()
        avg_iou_greatwall = torch.stack([x["IoU greatwall"] for x in outputs]).mean()
        if self.trainer.is_global_zero:
            self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU", avg_iou, self.current_epoch)
            self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU_non-greatwall", avg_iou_non_greatwall, self.current_epoch)
            self.logger.experiment.add_scalar("IoUs/" + train_or_val_or_test + "/IoU_greatwall", avg_iou_greatwall, self.current_epoch)

        if train_or_val_or_test == "Val":
            self.log('avg_metric_validation', avg_iou, sync_dist=True)
            self.log('avg_iou_non-greatwall_validation', avg_iou_non_greatwall, sync_dist=True)
            self.log('avg_iou_greatwall_validation', avg_iou_greatwall, sync_dist=True)

    # Pixel-level IoU accumulated across the batch.
    # def class_wise_iou(self, y_hat, y):
    #     """Compute mean and per-class IoU with ``absent_score=1.0`` semantics."""
    #     prediction = y_hat.argmax(dim=1)
    #     classwise_ious = []
    #     for class_index in range(self.num_classes):
    #         prediction_mask = prediction == class_index
    #         target_mask = y == class_index
    #         intersection = torch.logical_and(prediction_mask, target_mask).sum()
    #         union = torch.logical_or(prediction_mask, target_mask).sum()
    #         class_iou = torch.where(
    #             union > 0,
    #             intersection.float() / union.float(),
    #             torch.ones((), device=y_hat.device),
    #         )
    #         classwise_ious.append(class_iou)
    #     classwise_ious = torch.stack(classwise_ious)

    #     return [torch.mean(classwise_ious), *classwise_ious]

    # Macro IoU averaged over individual images.
    def class_wise_iou(self, y_hat, y):
        prediction = y_hat.argmax(dim=1)
        image_ious = []

        for image_index in range(prediction.shape[0]):
            class_ious = []

            for class_index in range(self.num_classes):
                prediction_mask = prediction[image_index] == class_index
                target_mask = y[image_index] == class_index

                intersection = torch.logical_and(
                    prediction_mask, target_mask
                ).sum()

                union = torch.logical_or(
                    prediction_mask, target_mask
                ).sum()

                class_iou = torch.where(
                    union > 0,
                    intersection.float() / union.float(),
                    torch.ones((), device=y_hat.device),
                )
                class_ious.append(class_iou)

            image_ious.append(torch.stack(class_ious))

        classwise_ious = torch.stack(image_ious).mean(dim=0)

        return [
            classwise_ious.mean(),
            *classwise_ious,
        ]


    def calc_loss(self, y_hat, y):
        """Combine segmentation loss and skeleton-guidance loss equally."""
        skel_width = 3
        mask_out, skel_out = y_hat
    
        # Main labels: [B, H, W], long, values 0 or 1.
        y_main = self.adapt_mask(y)

        # Generate skeleton supervision dynamically from each ground-truth mask.
        y_skel_main = generate_skeleton_edge(y_main, width=skel_width)
    
        # Segmentation loss: dynamically weighted CE + multiclass Dice.
        num_pos = torch.sum(y_main)
        num_neg = torch.sum(1.0 - y_main)
        dynamic_pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=100.0)
        class_weights = torch.stack([torch.ones_like(dynamic_pos_weight), dynamic_pos_weight])
        criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
        criterion_dice = DiceLoss("multiclass")
        loss_mask = (
            self.hparams.weight_loss * criterion_ce(mask_out, y_main)
            + (1 - self.hparams.weight_loss) * criterion_dice(mask_out, y_main)
        )
    
        # Skeleton loss.
        criterion_skel = SkeletonLoss()
        loss_skel = criterion_skel(skel_out, y_skel_main)
    
        # Preserve the original equal weighting between both tasks.
        total_loss = 0.5 * loss_mask + 0.5 * loss_skel
        return total_loss, self.class_wise_iou(mask_out, y_main)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SGGWSeg")
        parser.add_argument('--base_lr', default=3e-5, type=float)
        parser.add_argument('--max_lr', default=3e-4, type=float)
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument(
            '--pretrained_path',
            default='pretrained/SegMAN_Encoder_s.pth.tar',
            help='Path to the SegMAN-S ImageNet pretrained weights.',
        )

        # Hyperparameters for augmentation
        parser.add_argument('--bright', default=0., type=float)
        parser.add_argument('--wrap', default=0., type=float)
        parser.add_argument('--noise', default=0., type=float)
        parser.add_argument('--rotate', default=0., type=float)
        parser.add_argument('--hflip', default=0., type=float)
        parser.add_argument('--vflip', default=0., type=float)

        # Layer arguments
        parser.add_argument('--aspp', default=True, type=lambda x: (str(x).lower() == 'true'))

        # Loss arguments
        parser.add_argument('--weight_loss', default=0.5, type=float)

        return parent_parser
