import torch
from torch import nn
from typing import Tuple
from torch.nn import functional as F
from .metrics import bbox_iou, probiou  # Hàm tính IoU (Intersection over Union) và ProBIoU
from .tal import bbox2dist  # Hàm chuyển đổi tọa độ hộp giới hạn thành khoảng cách

def bbox_iou2(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    SIoU: bool = True,
    theta: float = 4.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if SIoU:
        # tính chi phí góc (angel cost)
        ch = torch.max(b1_x1, b2_x1) -torch. min(b1_x1, b2_x2)  # khoảng cách hai tâm theo trục x
        sigma = torch.sqrt((b1_x1 - b2_x2) ** 2 + (b1_y1 - b2_y2) ** 2)  # khoảng cách hai tâm hộp
        sin_theta = ch / sigma  # sin của góc giữa hai hộp
        angle = torch.arcsin(sin_theta.clamp(-1 + eps, 1 - eps))  # tránh NaN
        angle_cost = 1 - 2 * torch.sin(angle - torch.pi / 4) ** 2 # chi phí góc


        # tính chi phí khoảng cách (distance cost)
        cw = torch.max(b1_y1, b2_y1) - torch.min(b1_y1, b2_y2)  # khoảng cách hai tâm theo trục y
        h = torch.max(b1_x2, b2_x2) - torch.min(b1_x2, b2_x1)  # chiều cao của hộp bao quanh hai hộp
        w = torch.max(b1_y2, b2_y2) - torch.min(b1_y2, b2_y1)  # chiều rộng của hộp bao quanh hai hộp
        px = (ch/h)**2
        py = (cw/w)**2
        gamma = 2 - angle_cost  # gamma là hệ số điều chỉnh chi phí góc
        distance_cost = 2 - torch.exp(-gamma*px) - torch.exp(-gamma*py)  # chi phí khoảng cách

        # tính chi phí hình học (shape cost)
        omega_w = torch.abs(w1 - w2)/torch.max(w1, w2)  # chi phí hình học theo chiều rộng
        omega_h = torch.abs(h1 - h2)/torch.max(h1, h2)  # chi phí hình học theo chiều cao
        shape_cost = (1-torch.exp(-omega_w))**theta + (1-torch.exp(-omega_h))**theta  # chi phí hình học
        # Tính toán SIoU
        SIoU = 1 - iou + (distance_cost + shape_cost)/2
        return SIoU

    return iou  # IoU


class DFLoss(nn.Module):
    """Lớp tính toán Distribution Focal Loss (DFL) theo bài báo https://ieeexplore.ieee.org/document/9792391.
    DFL giúp cải thiện dự đoán tọa độ hộp giới hạn bằng cách phân phối xác suất trên các khoảng cách.
    """

    def __init__(self, reg_max: int = 16) -> None:
        """Khởi tạo lớp DFL với tham số reg_max.

        Args:
            reg_max (int): Giá trị tối đa của khoảng cách được biểu diễn (mặc định là 16).
        """
        super().__init__()  # Gọi hàm khởi tạo của lớp cha nn.Module
        self.reg_max = reg_max  # Lưu giá trị reg_max để sử dụng trong tính toán

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Tính toán tổng DFL losses (trái và phải) cho dự đoán khoảng cách.

        Args:
            pred_dist (torch.Tensor): Tensor dự đoán phân phối khoảng cách, shape [batch, num_anchors, reg_max].
            target (torch.Tensor): Tensor chứa khoảng cách mục tiêu, shape [batch, num_anchors].

        Returns:
            torch.Tensor: Tensor chứa DFL loss, shape [batch, num_anchors, 1].
        """
        # Giới hạn target trong khoảng [0, reg_max - 1 - 0.01] để tránh lỗi số học
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # Lấy chỉ số nguyên bên trái của target (target left)
        tr = tl + 1  # Chỉ số nguyên bên phải (target right)
        wl = tr - target  # Trọng số trái (khoảng cách từ target đến tr)
        wr = 1 - wl  # Trọng số phải (khoảng cách từ target đến tl)

        # Tính cross-entropy loss cho tl và tr, sau đó kết hợp với trọng số tương ứng
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tr.shape) * wr
        ).mean(-1, keepdim=True)  # Trung bình theo chiều cuối và giữ chiều cuối


class BboxLoss(nn.Module):
    """Lớp tính toán các loss liên quan đến hộp giới hạn (bounding box) trong bài toán phát hiện đối tượng.
    Bao gồm IoU loss và DFL loss (nếu được bật).
    """

    def __init__(self, reg_max: int = 16):
        """Khởi tạo lớp BboxLoss với tham số reg_max và khởi tạo DFL loss nếu cần.

        Args:
            reg_max (int): Giá trị tối đa của khoảng cách được biểu diễn (mặc định là 16).
        """
        super().__init__()  # Gọi hàm khởi tạo của lớp cha nn.Module
        # Khởi tạo DFL loss nếu reg_max > 1, ngược lại đặt là None
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tính toán IoU loss và DFL loss cho các hộp giới hạn.

        Args:
            pred_dist (torch.Tensor): Tensor dự đoán phân phối khoảng cách, shape [batch, num_anchors, reg_max].
            pred_bboxes (torch.Tensor): Tensor dự đoán tọa độ hộp giới hạn, shape [batch, num_anchors, 4].
            anchor_points (torch.Tensor): Tensor chứa tọa độ điểm neo, shape [num_anchors, 2].
            target_bboxes (torch.Tensor): Tensor chứa tọa độ hộp giới hạn mục tiêu, shape [batch, num_anchors, 4].
            target_scores (torch.Tensor): Tensor chứa điểm số mục tiêu, shape [batch, num_anchors, num_classes].
            target_scores_sum (torch.Tensor): Tổng điểm số mục tiêu, dùng để chuẩn hóa loss.
            fg_mask (torch.Tensor): Mặt nạ chỉ các hộp giới hạn thuộc foreground, shape [batch, num_anchors].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: IoU loss và DFL loss.
        """
        # Tính trọng số dựa trên tổng điểm số của các hộp foreground
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # Tính IoU loss cho các hộp foreground
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # Tính DFL loss nếu lớp DFL được khởi tạo
        if self.dfl_loss:
            # Chuyển tọa độ hộp mục tiêu thành khoảng cách (left, top, right, bottom)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            # Tính DFL loss cho các hộp foreground
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            # Nếu không có DFL loss, trả về tensor 0 trên cùng thiết bị với pred_dist
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
