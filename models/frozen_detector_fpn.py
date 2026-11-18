import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard mapping from Torchvision COCO 1-based category IDs to Ultralytics YOLO 0-based class IDs
COCO91_TO_YOLO80: Dict[int, int] = {
    1: 0,   # person
    2: 1,   # bicycle
    3: 2,   # car
    4: 3,   # motorcycle
    6: 5,   # bus
    7: 6,   # train
    8: 7,   # truck
}


class FrozenFPNDetectorTeacher(nn.Module):
    """
    Frozen Detector Teacher & Evaluator (`tscope-v3`, `F04` fix):
      - Training (`use_yolov8s=False` default): Single-teacher multi-scale FPN feature distillation
        using ONLY Frozen Faster R-CNN ResNet50-FPN v2 (`{p2, p3, p4, p5}`), keeping YOLOv8s,
        RetinaNet, FCOS, and SSDLite strictly unseen for true zero-shot cross-detector evaluation.
      - Evaluation (`use_yolov8s=True` optional): Computes exact globally pooled 101-point COCO AP@50
        and AP@[.50:.95] with MOT17/MOT20/KITTI ignore-region handling.
    """

    def __init__(self, pretrained: bool = True, use_yolov8s: bool = False) -> None:
        super().__init__()
        self.use_torchvision = False
        self.detector = None
        self.use_yolov8s = False
        self._yolo_wrapper_holder: List[Any] = []
        self.yolo_model = None

        if pretrained:
            try:
                from torchvision.models.detection import (
                    fasterrcnn_resnet50_fpn_v2,
                    FasterRCNN_ResNet50_FPN_V2_Weights,
                )

                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.detector = fasterrcnn_resnet50_fpn_v2(weights=weights)
                self.detector.eval()
                for p in self.detector.parameters():
                    p.requires_grad_(False)
                self.use_torchvision = True
            except Exception as e:
                print(f"[FrozenFPNDetectorTeacher] Fallback to lightweight pyramid ({e})")

            if use_yolov8s:
                try:
                    from ultralytics import YOLO

                    wrapper = YOLO("yolov8s.pt")
                    self._yolo_wrapper_holder = [wrapper]
                    self.yolo_model = wrapper.model
                    self.yolo_model.eval()
                    for p in self.yolo_model.parameters():
                        p.requires_grad_(False)
                    self.use_yolov8s = True
                except Exception as e:
                    print(f"[FrozenFPNDetectorTeacher] YOLOv8s unavailable ({e})")

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "yolo_wrapper":
            self._yolo_wrapper_holder = [] if value is None else [value]
            return
        super().__setattr__(name, value)

    @property
    def yolo_wrapper(self) -> Any:
        """Store Ultralytics YOLO outside nn.Module._modules so .eval() never triggers YOLO.train()."""
        return self._yolo_wrapper_holder[0] if self._yolo_wrapper_holder else None

    @yolo_wrapper.setter
    def yolo_wrapper(self, value: Any) -> None:
        self._yolo_wrapper_holder = [] if value is None else [value]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(dtype=x.dtype)) / self.std.to(dtype=x.dtype)

    def extract_yolo_backbone_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Optional YOLOv8s CSPDarknet multi-scale feature extractor {yolo_p3, yolo_p4, yolo_p5}."""
        if not self.use_yolov8s or self.yolo_model is None:
            return {}
        self.yolo_model.eval()
        first_param = next(self.yolo_model.parameters(), None)
        if first_param is not None and first_param.device != images.device:
            self.yolo_model.to(images.device)
        feats: Dict[str, torch.Tensor] = {}
        y_list: List[torch.Tensor] = []
        cur = torch.clamp(images, 0.0, 1.0)
        for idx, layer in enumerate(self.yolo_model.model[:10]):
            if layer.f != -1:
                cur = (
                    y_list[layer.f]
                    if isinstance(layer.f, int)
                    else [cur if j == -1 else y_list[j] for j in layer.f]
                )
            cur = layer(cur)
            y_list.append(cur)
            if idx == 4:
                feats["yolo_p3"] = cur
            elif idx == 6:
                feats["yolo_p4"] = cur
            elif idx == 9:
                feats["yolo_p5"] = cur
        return feats

    def _extract_resnet_fpn_tuple(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x_norm = self.normalize(images)
        feats = self.detector.backbone(x_norm)
        return tuple(
            feat for idx, (_, feat) in enumerate(feats.items()) if idx < 4
        )

    def _extract_yolo_fpn_tuple(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        y_feats = self.extract_yolo_backbone_features(images)
        return (y_feats["yolo_p3"], y_feats["yolo_p4"], y_feats["yolo_p5"])

    def extract_backbone_fpn_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full ResNet50-FPN v2 (+ optional YOLOv8s CSPDarknet when explicitly requested) feature extractor."""
        if self.use_torchvision and self.detector is not None:
            self.detector.eval()
            use_yolo = bool(self.use_yolov8s and self.yolo_model is not None)
            if torch.is_grad_enabled() and images.requires_grad:
                from torch.utils.checkpoint import checkpoint as grad_checkpoint

                chunk_size = 4
                n_imgs = images.shape[0]
                res_chunks = []
                yolo_chunks = []
                for start in range(0, n_imgs, chunk_size):
                    chunk = images[start : start + chunk_size]
                    res_chunks.append(
                        grad_checkpoint(self._extract_resnet_fpn_tuple, chunk, use_reentrant=False)
                    )
                    if use_yolo:
                        yolo_chunks.append(
                            grad_checkpoint(self._extract_yolo_fpn_tuple, chunk, use_reentrant=False)
                        )
                num_res = len(res_chunks[0])
                out = {
                    f"p{idx + 2}": (
                        res_chunks[0][idx]
                        if len(res_chunks) == 1
                        else torch.cat([c[idx] for c in res_chunks], dim=0)
                    )
                    for idx in range(num_res)
                }
                if use_yolo:
                    for idx, name in enumerate(("yolo_p3", "yolo_p4", "yolo_p5")):
                        out[name] = (
                            yolo_chunks[0][idx]
                            if len(yolo_chunks) == 1
                            else torch.cat([c[idx] for c in yolo_chunks], dim=0)
                        )
                return out

            x_norm = self.normalize(images)
            feats = self.detector.backbone(x_norm)
            out = {
                f"p{idx + 2}": feat
                for idx, (_, feat) in enumerate(feats.items())
                if idx < 4
            }
            if use_yolo:
                out.update(self.extract_yolo_backbone_features(images))
            return out
        return self.extract_fast_pyramid(images)

    def extract_fast_pyramid(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fast differentiable multi-scale spatial+gradient pyramid."""
        x_norm = self.normalize(images)
        dx = x_norm[..., :, 1:] - x_norm[..., :, :-1]
        dy = x_norm[..., 1:, :] - x_norm[..., :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        feat_base = torch.cat([x_norm, dx, dy], dim=1)
        return {
            "p2": F.avg_pool2d(feat_base, kernel_size=4, stride=4),
            "p3": F.avg_pool2d(feat_base, kernel_size=8, stride=8),
            "p4": F.avg_pool2d(feat_base, kernel_size=16, stride=16),
        }

    def extract_fpn_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.extract_backbone_fpn_features(images)

    @staticmethod
    def _extract_connected_boxes_from_mask(mask_2d: torch.Tensor) -> torch.Tensor:
        ys, xs = torch.where(mask_2d > 0.5)
        if len(ys) == 0:
            return torch.zeros((0, 4), device=mask_2d.device, dtype=torch.float32)
        return torch.tensor(
            [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]],
            device=mask_2d.device,
            dtype=torch.float32,
        )

    @staticmethod
    def _prepare_dataset_ap_cache(
        all_gt_boxes: List[torch.Tensor],
        all_pred_boxes: List[torch.Tensor],
        all_pred_scores: List[torch.Tensor],
        all_ignore_boxes: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[int, List[int], List[Optional[np.ndarray]], List[Optional[np.ndarray]], List[Tuple[int, int]]]:
        """Validate frame tensors and precompute per-image IoU matrices and sorted prediction order."""
        from torchvision.ops import box_iou

        count = len(all_gt_boxes)
        if (
            len(all_pred_boxes) != count
            or len(all_pred_scores) != count
            or (all_ignore_boxes is not None and len(all_ignore_boxes) != count)
        ):
            raise ValueError("AP requires aligned GT, prediction, score, and ignore frame lists")
        for index in range(count):
            boxes = [all_gt_boxes[index], all_pred_boxes[index]]
            if all_ignore_boxes is not None:
                boxes.append(all_ignore_boxes[index])
            for tensor in boxes:
                if (
                    tensor.ndim != 2
                    or tensor.shape[1] != 4
                    or not bool(torch.isfinite(tensor).all())
                    or bool(((tensor[:, 2:] - tensor[:, :2]) <= 0).any())
                ):
                    raise ValueError("AP box tensors must contain finite nondegenerate XYXY boxes")
            scores = all_pred_scores[index]
            if (
                scores.ndim != 1
                or scores.numel() != all_pred_boxes[index].shape[0]
                or not bool(torch.isfinite(scores).all())
                or bool(((scores < 0) | (scores > 1)).any())
            ):
                raise ValueError("AP confidence scores must align with boxes and lie in [0, 1]")

        total_gt = sum(int(g.shape[0]) for g in all_gt_boxes)
        if total_gt == 0:
            return 0, [], [], [], []

        n_gts: List[int] = []
        img_gt_ious: List[Optional[np.ndarray]] = []
        img_ig_max_ious: List[Optional[np.ndarray]] = []
        flat_preds: List[Tuple[float, int, int]] = []
        for img_idx, (pb, ps) in enumerate(zip(all_pred_boxes, all_pred_scores)):
            pb_cpu = pb.detach().cpu()
            ps_cpu = ps.detach().cpu()
            gb_cpu = all_gt_boxes[img_idx].detach().cpu()
            n_gts.append(int(gb_cpu.shape[0]))
            if pb_cpu.shape[0] > 0 and gb_cpu.shape[0] > 0:
                img_gt_ious.append(box_iou(pb_cpu, gb_cpu).numpy())
            else:
                img_gt_ious.append(None)
            if (
                all_ignore_boxes is not None
                and img_idx < len(all_ignore_boxes)
                and pb_cpu.shape[0] > 0
                and all_ignore_boxes[img_idx].shape[0] > 0
            ):
                ig_cpu = all_ignore_boxes[img_idx].detach().cpu()
                img_ig_max_ious.append(box_iou(pb_cpu, ig_cpu).max(dim=1).values.numpy())
            else:
                img_ig_max_ious.append(None)
            for j, sc in enumerate(ps_cpu.tolist()):
                flat_preds.append((float(sc), img_idx, j))

        flat_preds.sort(key=lambda x: x[0], reverse=True)
        sorted_pairs = [(img_idx, pred_j) for _, img_idx, pred_j in flat_preds]
        return total_gt, n_gts, img_gt_ious, img_ig_max_ious, sorted_pairs

    @staticmethod
    def _score_dataset_ap_at_threshold(
        total_gt: int,
        n_gts: List[int],
        img_gt_ious: List[Optional[np.ndarray]],
        img_ig_max_ious: List[Optional[np.ndarray]],
        sorted_pairs: List[Tuple[int, int]],
        iou_thresh: float,
    ) -> float:
        """Compute exact 101-point COCO AP for a single IoU threshold using precomputed IoU matrices."""
        if total_gt == 0:
            return float("nan")
        if not sorted_pairs:
            return 0.0

        gt_matched = [np.zeros(n, dtype=bool) for n in n_gts]
        neg_one = np.float32(-1.0)
        tp_list: List[float] = []
        fp_list: List[float] = []

        for img_idx, pred_j in sorted_pairs:
            gt_iou_mat = img_gt_ious[img_idx]
            matched_active = False
            if gt_iou_mat is not None:
                m_flags = gt_matched[img_idx]
                if not m_flags.all():
                    masked = np.where(m_flags, neg_one, gt_iou_mat[pred_j])
                    best_gt_idx = int(np.argmax(masked))
                    if float(masked[best_gt_idx]) >= iou_thresh:
                        m_flags[best_gt_idx] = True
                        tp_list.append(1.0)
                        fp_list.append(0.0)
                        matched_active = True
            if matched_active:
                continue

            ig_max = img_ig_max_ious[img_idx]
            if ig_max is not None and float(ig_max[pred_j]) >= iou_thresh:
                continue

            tp_list.append(0.0)
            fp_list.append(1.0)

        if not tp_list:
            return 0.0

        tp = np.asarray(tp_list, dtype=np.float64)
        fp = np.asarray(fp_list, dtype=np.float64)
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall = tp_cum / float(total_gt)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        mpre = np.maximum.accumulate(precision[::-1])[::-1]

        recall_grid = np.linspace(0.0, 1.0, 101, dtype=np.float64)
        ap_vals = np.zeros_like(recall_grid)
        inds = np.searchsorted(recall, recall_grid, side="left")
        valid = inds < recall.shape[0]
        ap_vals[valid] = mpre[inds[valid]]
        return float(ap_vals.mean())

    @staticmethod
    def compute_dataset_level_ap(
        all_gt_boxes: List[torch.Tensor],
        all_pred_boxes: List[torch.Tensor],
        all_pred_scores: List[torch.Tensor],
        all_ignore_boxes: Optional[List[torch.Tensor]] = None,
        iou_thresh: float = 0.50,
    ) -> float:
        if not math.isfinite(iou_thresh) or not 0.0 < iou_thresh <= 1.0:
            raise ValueError("Invalid AP IoU threshold")
        total_gt, n_gts, img_gt_ious, img_ig_max_ious, sorted_pairs = (
            FrozenFPNDetectorTeacher._prepare_dataset_ap_cache(
                all_gt_boxes,
                all_pred_boxes,
                all_pred_scores,
                all_ignore_boxes=all_ignore_boxes,
            )
        )
        return FrozenFPNDetectorTeacher._score_dataset_ap_at_threshold(
            total_gt, n_gts, img_gt_ious, img_ig_max_ious, sorted_pairs, iou_thresh
        )

    @staticmethod
    def compute_dataset_level_coco_metrics(
        all_gt_boxes: List[torch.Tensor],
        all_pred_boxes: List[torch.Tensor],
        all_pred_scores: List[torch.Tensor],
        all_ignore_boxes: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        total_gt, n_gts, img_gt_ious, img_ig_max_ious, sorted_pairs = (
            FrozenFPNDetectorTeacher._prepare_dataset_ap_cache(
                all_gt_boxes,
                all_pred_boxes,
                all_pred_scores,
                all_ignore_boxes=all_ignore_boxes,
            )
        )
        if total_gt == 0:
            return {"gt_ap50_pct": float("nan"), "gt_ap50_95_pct": float("nan")}
        iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
        aps = [
            FrozenFPNDetectorTeacher._score_dataset_ap_at_threshold(
                total_gt, n_gts, img_gt_ious, img_ig_max_ious, sorted_pairs, thr
            )
            for thr in iou_thresholds
        ]
        if any(math.isnan(x) for x in aps):
            return {"gt_ap50_pct": float("nan"), "gt_ap50_95_pct": float("nan")}
        return {
            "gt_ap50_pct": round(float(aps[0] * 100.0), 3),
            "gt_ap50_95_pct": round(float(sum(aps) / len(aps) * 100.0), 3),
        }

    @staticmethod
    def compute_multiclass_dataset_coco_metrics(
        all_gt_boxes: List[torch.Tensor],
        all_gt_classes: List[torch.Tensor],
        all_pred_boxes: List[torch.Tensor],
        all_pred_scores: List[torch.Tensor],
        all_pred_classes: List[torch.Tensor],
        all_ignore_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Category-wise 101-point COCO mAP@50 and mAP@[.50:.95] across multi-class VCM benchmarks
        (`KITTI`, `SFU-HW-Objects-v1`), averaging over all classes that appear in GT.
        """
        present = sorted(
            {
                int(c)
                for cls_t in all_gt_classes
                for c in cls_t.detach().cpu().tolist()
                if int(c) > 0 and (target_classes is None or int(c) in target_classes)
            }
        )
        if not present:
            return {"gt_ap50_pct": float("nan"), "gt_ap50_95_pct": float("nan"), "per_class": {}}

        ap50_list: List[float] = []
        ap50_95_list: List[float] = []
        per_cls: Dict[int, Dict[str, float]] = {}
        for cid in present:
            gt_b_c = [
                gb[gc == cid] if gb.shape[0] > 0 else gb
                for gb, gc in zip(all_gt_boxes, all_gt_classes)
            ]
            pr_b_c = [
                pb[pc == cid] if pb.shape[0] > 0 else pb
                for pb, pc in zip(all_pred_boxes, all_pred_classes)
            ]
            pr_s_c = [
                ps[pc == cid] if ps.shape[0] > 0 else ps
                for ps, pc in zip(all_pred_scores, all_pred_classes)
            ]
            res_c = FrozenFPNDetectorTeacher.compute_dataset_level_coco_metrics(
                gt_b_c, pr_b_c, pr_s_c, all_ignore_boxes=all_ignore_boxes
            )
            if not math.isnan(res_c["gt_ap50_pct"]):
                ap50_list.append(res_c["gt_ap50_pct"])
                ap50_95_list.append(res_c["gt_ap50_95_pct"])
                per_cls[cid] = res_c

        if not ap50_list:
            return {"gt_ap50_pct": float("nan"), "gt_ap50_95_pct": float("nan"), "per_class": {}}
        return {
            "gt_ap50_pct": round(float(np.mean(ap50_list)), 3),
            "gt_ap50_95_pct": round(float(np.mean(ap50_95_list)), 3),
            "per_class": per_cls,
        }

    @staticmethod
    def _compute_ap_at_iou(
        gt_boxes: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        iou_thresh: float = 0.5,
        ignore_boxes: Optional[torch.Tensor] = None,
    ) -> float:
        ig_list = [ignore_boxes] if ignore_boxes is not None else None
        return FrozenFPNDetectorTeacher.compute_dataset_level_ap(
            [gt_boxes], [pred_boxes], [pred_scores], all_ignore_boxes=ig_list, iou_thresh=iou_thresh
        )

    @torch.no_grad()
    def evaluate_task_accuracy(
        self,
        clean_video: torch.Tensor,
        recon_video: torch.Tensor,
        m_det: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        ignore_boxes: Optional[torch.Tensor] = None,
        require_real_detector: bool = False,
        return_raw_detections: bool = False,
        cached_clean_info: Optional[Dict[str, Any]] = None,
        gt_classes: Optional[torch.Tensor] = None,
        target_coco_classes: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        if require_real_detector and (not self.use_torchvision or self.detector is None):
            raise RuntimeError(
                "Gate 0.5 violation: Real torchvision Faster R-CNN detector weights are required for AP evaluation."
            )

        active_coco_classes = tuple(target_coco_classes) if target_coco_classes is not None else (1,)
        b, t, c, h, w = clean_video.shape
        clean_flat = clean_video.view(b * t, c, h, w)
        recon_flat = recon_video.view(b * t, c, h, w)
        m_det_flat = m_det.view(b * t, 1, h, w)

        t_feats = (
            cached_clean_info["t_feats"]
            if cached_clean_info is not None and "t_feats" in cached_clean_info
            else self.extract_fast_pyramid(clean_flat)
        )
        s_feats = self.extract_fast_pyramid(recon_flat)

        cos_scores: List[float] = []
        for k, f_t in t_feats.items():
            f_s = s_feats[k]
            m_l = F.interpolate(m_det_flat, size=f_t.shape[-2:], mode="nearest")
            if float(m_l.sum().item()) > 0:
                cos_map = F.cosine_similarity(f_t.to(f_s.device), f_s, dim=1, eps=1e-6).unsqueeze(1)
                masked_cos = (m_l * cos_map).sum() / (m_l.sum() + 1e-6)
                cos_scores.append(float(masked_cos.item()) * 100.0)

        fpn_cos_pct = float(sum(cos_scores) / len(cos_scores)) if cos_scores else float("nan")

        if float(m_det.sum().item()) > 0:
            mse_obj = ((m_det * (clean_video - recon_video)) ** 2).sum() / (
                m_det.sum() * 3.0 + 1e-6
            )
            psnr_obj = float(10.0 * torch.log10(1.0 / (mse_obj + 1e-8)).item())
        else:
            mse_all = ((clean_video - recon_video) ** 2).mean()
            psnr_obj = float(10.0 * torch.log10(1.0 / (mse_all + 1e-8)).item())

        gt_ap50_pct = float("nan")
        gt_ap50_95_pct = float("nan")
        teacher_ap50_pct = float("nan")
        yolo_gt_ap50_pct = float("nan")
        yolo_gt_ap50_95_pct = float("nan")

        clip_gt_boxes: List[torch.Tensor] = []
        clip_gt_classes: List[torch.Tensor] = []
        clip_ig_boxes: List[torch.Tensor] = []
        clip_teacher_boxes: List[torch.Tensor] = []
        clip_pred_boxes: List[torch.Tensor] = []
        clip_pred_scores: List[torch.Tensor] = []
        clip_pred_classes: List[torch.Tensor] = []
        clip_yolo_pred_boxes: List[torch.Tensor] = []
        clip_yolo_pred_scores: List[torch.Tensor] = []
        clip_yolo_pred_classes: List[torch.Tensor] = []

        if self.use_torchvision and self.detector is not None:
            self.detector.eval()
            det_chunk_size = 4
            if cached_clean_info is not None and "clean_dets" in cached_clean_info:
                clean_dets = cached_clean_info["clean_dets"]
            else:
                clean_dets = []
                for st in range(0, clean_flat.shape[0], det_chunk_size):
                    sub_c = self.detector(list(torch.clamp(clean_flat[st : st + det_chunk_size], 0.0, 1.0)))
                    clean_dets.extend({k: v.detach().cpu() for k, v in d.items()} for d in sub_c)
            recon_dets = []
            for st in range(0, recon_flat.shape[0], det_chunk_size):
                sub_r = self.detector(list(torch.clamp(recon_flat[st : st + det_chunk_size], 0.0, 1.0)))
                recon_dets.extend({k: v.detach().cpu() for k, v in d.items()} for d in sub_r)

            gt_flat = gt_boxes.view(b * t, -1, 4) if gt_boxes is not None else None
            gt_cls_flat = gt_classes.view(b * t, -1) if gt_classes is not None else None
            ig_flat = ignore_boxes.view(b * t, -1, 4) if ignore_boxes is not None else None

            allowed_labels = torch.tensor(active_coco_classes, dtype=torch.long)
            for f_idx, (c_det, r_det) in enumerate(zip(clean_dets, recon_dets)):
                labels_r_cpu = r_det["labels"].detach().cpu()
                scores_r_cpu = r_det["scores"].detach().cpu()
                boxes_r_cpu = r_det["boxes"].detach().cpu()
                cls_mask_r = torch.isin(labels_r_cpu, allowed_labels) & (scores_r_cpu >= 0.01)
                clip_pred_boxes.append(boxes_r_cpu[cls_mask_r])
                clip_pred_scores.append(scores_r_cpu[cls_mask_r])
                clip_pred_classes.append(labels_r_cpu[cls_mask_r])

                if gt_flat is not None:
                    f_gt = gt_flat[f_idx].detach().cpu()
                    valid_mask = f_gt[:, 0] >= 0.0
                    valid_gt = f_gt[valid_mask]
                    clip_gt_boxes.append(valid_gt)
                    if gt_cls_flat is not None:
                        f_gc = gt_cls_flat[f_idx].detach().cpu()
                        clip_gt_classes.append(f_gc[valid_mask].long())
                    else:
                        clip_gt_classes.append(torch.ones((valid_gt.shape[0],), dtype=torch.long))
                else:
                    clip_gt_boxes.append(torch.zeros((0, 4), dtype=torch.float32))
                    clip_gt_classes.append(torch.zeros((0,), dtype=torch.long))

                if ig_flat is not None:
                    f_ig = ig_flat[f_idx].detach().cpu()
                    valid_ig = f_ig[f_ig[:, 0] >= 0.0]
                    clip_ig_boxes.append(valid_ig)
                else:
                    clip_ig_boxes.append(torch.zeros((0, 4), dtype=torch.float32))

                labels_c_cpu = c_det["labels"].detach().cpu()
                scores_c_cpu = c_det["scores"].detach().cpu()
                boxes_c_cpu = c_det["boxes"].detach().cpu()
                cls_mask_c = torch.isin(labels_c_cpu, allowed_labels) & (scores_c_cpu >= 0.50)
                clip_teacher_boxes.append(boxes_c_cpu[cls_mask_c])

            if gt_flat is not None:
                if len(active_coco_classes) == 1:
                    coco_gt = self.compute_dataset_level_coco_metrics(
                        clip_gt_boxes, clip_pred_boxes, clip_pred_scores, all_ignore_boxes=clip_ig_boxes
                    )
                else:
                    coco_gt = self.compute_multiclass_dataset_coco_metrics(
                        clip_gt_boxes,
                        clip_gt_classes,
                        clip_pred_boxes,
                        clip_pred_scores,
                        clip_pred_classes,
                        all_ignore_boxes=clip_ig_boxes,
                        target_classes=active_coco_classes,
                    )
                gt_ap50_pct = coco_gt["gt_ap50_pct"]
                gt_ap50_95_pct = coco_gt["gt_ap50_95_pct"]

            t_ap = self.compute_dataset_level_ap(
                clip_teacher_boxes, clip_pred_boxes, clip_pred_scores, iou_thresh=0.50
            )
            if not math.isnan(t_ap):
                teacher_ap50_pct = round(float(t_ap * 100.0), 3)

            if self.use_yolov8s and self.yolo_wrapper is not None:
                yolo_classes = [
                    COCO91_TO_YOLO80[c_id]
                    for c_id in active_coco_classes
                    if c_id in COCO91_TO_YOLO80
                ] or [0]
                yolo_to_coco = {v: k for k, v in COCO91_TO_YOLO80.items()}
                yolo_imgsz = (h, w) if (h % 32 == 0 and w % 32 == 0) else max(h, w)
                yolo_res = []
                for st_y in range(0, recon_flat.shape[0], 8):
                    yolo_res.extend(
                        self.yolo_wrapper.predict(
                            source=torch.clamp(recon_flat[st_y : st_y + 8], 0.0, 1.0),
                            imgsz=yolo_imgsz,
                            conf=0.01,
                            classes=yolo_classes,
                            verbose=False,
                        )
                    )
                for r_y in yolo_res:
                    clip_yolo_pred_boxes.append(r_y.boxes.xyxy.detach().cpu())
                    clip_yolo_pred_scores.append(r_y.boxes.conf.detach().cpu())
                    y_cls = r_y.boxes.cls.detach().cpu().long()
                    mapped_cls = torch.tensor(
                        [yolo_to_coco.get(int(v.item()), 1) for v in y_cls], dtype=torch.long
                    )
                    clip_yolo_pred_classes.append(mapped_cls)
                if gt_flat is not None:
                    if len(active_coco_classes) == 1:
                        coco_yolo = self.compute_dataset_level_coco_metrics(
                            clip_gt_boxes,
                            clip_yolo_pred_boxes,
                            clip_yolo_pred_scores,
                            all_ignore_boxes=clip_ig_boxes,
                        )
                    else:
                        coco_yolo = self.compute_multiclass_dataset_coco_metrics(
                            clip_gt_boxes,
                            clip_gt_classes,
                            clip_yolo_pred_boxes,
                            clip_yolo_pred_scores,
                            clip_yolo_pred_classes,
                            all_ignore_boxes=clip_ig_boxes,
                            target_classes=active_coco_classes,
                        )
                    yolo_gt_ap50_pct = coco_yolo["gt_ap50_pct"]
                    yolo_gt_ap50_95_pct = coco_yolo["gt_ap50_95_pct"]

        out: Dict[str, Any] = {
            "gt_ap50_pct": gt_ap50_pct,
            "gt_ap50_95_pct": gt_ap50_95_pct,
            "yolo_gt_ap50_pct": yolo_gt_ap50_pct,
            "yolo_gt_ap50_95_pct": yolo_gt_ap50_95_pct,
            "teacher_ap50_pct": teacher_ap50_pct,
            "map50_pct": gt_ap50_pct,
            "map50_95_pct": gt_ap50_95_pct,
            "fpn_cos_sim_pct": round(fpn_cos_pct, 3) if not math.isnan(fpn_cos_pct) else float("nan"),
            "task_roi_psnr_db": round(psnr_obj, 3),
        }
        if return_raw_detections:
            raw_dict: Dict[str, Any] = {
                "gt_boxes": clip_gt_boxes,
                "gt_classes": clip_gt_classes,
                "ignore_boxes": clip_ig_boxes,
                "teacher_boxes": clip_teacher_boxes,
                "pred_boxes": clip_pred_boxes,
                "pred_scores": clip_pred_scores,
                "pred_classes": clip_pred_classes,
            }
            if clip_yolo_pred_boxes:
                raw_dict["yolo_pred_boxes"] = clip_yolo_pred_boxes
                raw_dict["yolo_pred_scores"] = clip_yolo_pred_scores
                raw_dict["yolo_pred_classes"] = clip_yolo_pred_classes
            out["raw_detections"] = raw_dict
        return out
