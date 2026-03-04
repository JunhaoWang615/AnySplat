import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class PoseLoss(nn.Module):
    def __init__(self, rotation_weight=1.0):
        super().__init__()
        self.rotation_weight = rotation_weight

    @staticmethod
    def matrix_to_quaternion(r_mat):
        """旋转矩阵转四元数 (w, x, y, z)"""
        m00, m01, m02 = r_mat[..., 0, 0], r_mat[..., 0, 1], r_mat[..., 0, 2]
        m10, m11, m12 = r_mat[..., 1, 0], r_mat[..., 1, 1], r_mat[..., 1, 2]
        m20, m21, m22 = r_mat[..., 2, 0], r_mat[..., 2, 1], r_mat[..., 2, 2]

        q_w = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-6)) / 2
        q_x = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-6)) / 2
        q_y = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-6)) / 2
        q_z = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-6)) / 2

        q_x = torch.where(m21 - m12 > 0, q_x, -q_x)
        q_y = torch.where(m02 - m20 > 0, q_y, -q_y)
        q_z = torch.where(m10 - m01 > 0, q_z, -q_z)
        return torch.stack([q_w, q_x, q_y, q_z], dim=-1)

    def forward(self, pred_pose, gt_pose):
        """
        pred_pose, gt_pose: [B, N, 4, 4]
        """
        # 平移损失 (MSE)
        t_loss = F.mse_loss(pred_pose[..., :3, 3], gt_pose[..., :3, 3])
        
        t_loss = math.log(1 + t_loss, 20)
        
        # 旋转损失 (基于四元数内积)
        q_pred = self.matrix_to_quaternion(pred_pose[..., :3, :3])
        q_gt = self.matrix_to_quaternion(gt_pose[..., :3, :3])
        
        # 1 - <q1, q2>^2，解决 q 和 -q 的双重覆盖问题
        inner_prod = torch.sum(q_pred * q_gt, dim=-1)
        r_loss = 1 - torch.mean(inner_prod**2)
        
        total_loss = t_loss + self.rotation_weight * r_loss
        return total_loss, t_loss, r_loss