import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model.GeometricExtractor import CurvatureComputer

def dual_softmax_loss(X, Y, temp = 0.2):
    """
    Dual Softmax Loss (双重 Softmax 损失)
    
    功能:
        用于计算特征描述子之间的匹配损失。它通过在两个方向（X->Y 和 Y->X）上分别应用 Softmax
        来计算匹配概率，并结合 NLLLoss 进行优化。这一过程强化了互为最近邻（Mutual Nearest Neighbor）的约束。
        
    公式:
        Dist(i, j) = <X_i, Y_j> / temp
        P(j|i) = softmax_j(Dist(i, j))  [X -> Y 的匹配概率]
        P(i|j) = softmax_i(Dist(i, j))  [Y -> X 的匹配概率]
        Confidence(i, j) = P(j|i) * P(i|j)
        Loss = NLLLoss(log(P(j|i)), target) + NLLLoss(log(P(i|j)), target)
        
    参数:
        X (Tensor): 图像1的特征描述子 (B*N, D)
        Y (Tensor): 图像2的特征描述子 (B*N, D)
        temp (float): 温度系数，用于缩放相似度，控制分布的尖锐程度。
    """
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp( conf_matrix12 ).max(dim=-1)[0]
        conf21 = torch.exp( conf_matrix21 ).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device = X.device)

    loss = F.nll_loss(conf_matrix12, target) + F.nll_loss(conf_matrix21, target)

    return loss, conf


class LogCoshCircleLoss(nn.Module):
    """
    Log-Cosh Circle Loss (对数双曲余弦 Circle 损失)
    
    功能:
        这是一种专为度量学习设计的改进型损失函数，结合了 Circle Loss 的难样本挖掘能力和
        Log-Cosh 的平滑鲁棒性。
        1. Circle Loss: 动态加权正负样本的梯度，自动聚焦于难以区分的样本对（相似度低的正样本，相似度高的负样本）。
        2. Log-Cosh: log(cosh(x)) 是一种类似于 Huber Loss 的平滑函数。在误差较小时表现为 L2 (x^2/2)，
           在误差较大时表现为 L1 (|x| - log2)。这使得模型对噪声和异常值（Outliers）更加鲁棒。
    
    公式:
        Logit_p = -ap * (sp - delta_p) * gamma
        Logit_n = an * (sn - delta_n) * gamma
        L_circle = softplus( logsumexp(Logit_n) + logsumexp(Logit_p) )
        L_final = log( cosh( L_circle ) )
        
        其中:
        sp: 正样本对相似度, sn: 负样本对相似度
        ap = clamp(1 + m - sp), an = clamp(sn + m) [自适应权重]
        delta_p = 1 - m, delta_n = m [边际阈值]
    
    参数:
        m (float): Margin (边际)，用于控制正负样本的分离程度。默认 0.25。
        gamma (float): Scale factor (缩放因子)，用于放大梯度。默认 64。
    """
    def __init__(self, m: float = 0.25, gamma: float = 64) -> None:
        super(LogCoshCircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        # sp: (N) or (B, 1) - Similarity of Positive pairs
        # sn: (N) or (B, M) - Similarity of Negative pairs
        
        # Ensure dimensionality
        if sp.dim() == 1: sp = sp.unsqueeze(1)
        if sn.dim() == 1: sn = sn.unsqueeze(1)
        
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        # Sum over negative samples (dim=1) and positive samples (dim=1)
        # For circle loss with 1 positive per anchor, logsumexp(logit_p) is just logit_p[0]
        loss_circle = self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))
        
        # Log-Cosh Smoothing (element-wise first, then mean)
        loss = torch.log(torch.cosh(loss_circle))
        
        return loss.mean()

class GeoFeatLoss(nn.Module):
    """
    GeoFeat Losses (GeoFeat 总体损失函数)
    
    功能:
        封装了特征检测、描述子匹配、以及几何一致性（Geometric Consistency）约束的所有损失函数。
        
    包含:
        - 描述子匹配损失 (Descriptor Loss): 使用 LogCoshCircleLoss 计算描述子相似度误差，
          加强大视差下的匹配鲁棒性。
        - 关键点损失 (Keypoint Loss): 对关键点提取结果进行自适应蒸馏（Distillation）。
          使用 ALike 方法作为 pseudo-GT 进行监督。
        - 几何特征损失 (Geometric Loss): 控制深度、法向量、梯度、曲率等几何信息的一致性。
        - 坐标回归损失 (Coordinate Loss): 直接监督特征点的位置坐标预测精度。
        - 深层监督损失 (Deep Supervision): 对中间层特征进行额外的辅助监督，加速收敛。
        
    公式:
        L_total = lambda_descs * L_circle(D1, D2) 
                  + lambda_kpts * L_distill(K1, K2)
                  + lambda_geo * (L_depth + L_normal + L_curv + ...)
                  ...
    
    参数:
        lam_*: 各个子损失项的权重系数。
        geo_config: 控制哪些几何特征参与训练 (normal, depth, etc.)
    """
    def __init__(self,
                device,
                lam_descs=1,
                lam_fb_descs=1,
                lam_kpts=1,
                lam_heatmap=1,
                lam_normals=1,
                lam_coordinates=1,
                lam_fb_coordinates=1,
                lam_depth=1,
                lam_gradients=1,
                lam_curvature=1,
                lam_deep_supervision=1.0,
                depth_spvs=True, 
                geo_config=None):
        super().__init__()
        
        # loss parameters (损失函数权重参数)
        self.lam_descs=lam_descs                  # 初始描述子匹配损失权重
        self.lam_fb_descs=lam_fb_descs            # 增强描述子匹配损失权重 (Feature Booster)
        self.lam_kpts=lam_kpts                    # 关键点检测损失权重
        self.lam_heatmap=lam_heatmap              # 热力图损失权重
        self.lam_normals=lam_normals              # 法向量一致性损失权重
        self.lam_coordinates=lam_coordinates      # 初始特征坐标回归损失权重
        self.lam_fb_coordinates=lam_fb_coordinates # 增强特征坐标回归损失权重
        self.lam_depth=lam_depth                  # 深度估计损失权重
        self.lam_gradients=lam_gradients          # 深度梯度损失权重
        self.lam_curvature=lam_curvature          # 曲率一致性损失权重
        self.lam_deep_supervision=lam_deep_supervision # 深度监督损失权重 (中间层监督)
        self.depth_spvs=depth_spvs                # 是否启用深度监督
        self.geo_config = geo_config if geo_config is not None else {}
        
        # 运行时损失记录变量
        self.running_descs_loss=0
        self.running_kpts_loss=0
        self.running_heatmaps_loss=0
        
        # 当前批次损失值
        self.loss_descs=0           # 描述子损失
        self.loss_fb_descs=0        # 增强描述子损失
        self.loss_kpts=0            # 关键点损失
        self.loss_heatmaps=0        # 热力图损失
        self.loss_normals=0         # 法向量损失
        self.loss_coordinates=0     # 坐标损失
        self.loss_fb_coordinates=0  # 增强坐标损失
        self.loss_depths=0          # 深度损失
        self.loss_gradients=0       # 梯度损失
        self.loss_curvature=0       # 曲率损失
        self.loss_deep_supervision=0 # 深度监督损失
        
        # 当前批次精度指标
        self.acc_coarse=0           # 粗匹配精度
        self.acc_fb_coarse=0        # 增强粗匹配精度
        self.acc_kpt=0              # 关键点精度
        self.acc_coordinates=0      # 坐标回归精度
        self.acc_fb_coordinates=0   # 增强坐标回归精度
        
        # device
        self.dev=device
        self.curvature_computer = CurvatureComputer().to(device)
        self.circle_loss = LogCoshCircleLoss(m=0.25, gamma=64).to(device)
        
    
    def check_accuracy(self,m1,m2,pts1=None,pts2=None,plot=False):
        with torch.no_grad():
            #dist_mat = torch.cdist(X,Y)
            dist_mat = m1 @ m2.t()
            nn = torch.argmax(dist_mat, dim=1)
            #nn = torch.argmin(dist_mat, dim=1)
            correct = nn == torch.arange(len(m1), device = m1.device)

            if pts1 is not None and plot:
                import matplotlib.pyplot as plt
                canvas = torch.zeros((60, 80),device=m1.device)
                pts1 = pts1[~correct]
                canvas[pts1[:,1].long(), pts1[:,0].long()] = 1
                canvas = canvas.cpu().numpy()
                plt.imshow(canvas), plt.show()

            acc = correct.sum().item() / len(m1)
            return acc    
    
    def compute_descriptors_loss(self,descs1,descs2,pts):
        """
        Descriptor Loss (Log-Cosh Circle Loss)
        
        功能:
            计算描述子对（Pairs）之间的 Circle Loss，并同时返回粗匹配准确率和置信度矩阵。
            用于拉近所有正样本对（Matched Pairs）在特征空间的余弦距离，并推远负样本对（Unmatched Pairs）。
            
        过程:
            1. 取出批次中对应的描述子 m1 和 m2。
            2. 使用 m1 @ m2.T 构建相似度矩阵。
            3. 对角线为正样本 (Similarity of Positive pairs, sp)，非对角线为负样本 (Similarity of Negative pairs, sn)。
            4. 将 sp, sn 输入 LogCoshCircleLoss 计算每一行的 Circle Loss。
            5. 同时使用 m1 @ m2.T 计算双重 Softmax 损失的 confidence (用于后续热力图/坐标监督)。
        """
        loss=[]
        acc=0
        B,_,H,W=descs1.shape
        conf_list=[]
        
        for b in range(B):
            pts1,pts2=pts[b][:,:2],pts[b][:,2:]
            m1=descs1[b,:,pts1[:,1].long(),pts1[:,0].long()].permute(1,0)
            m2=descs2[b,:,pts2[:,1].long(),pts2[:,0].long()].permute(1,0)
            
            # --- New Logic: Log-Cosh Circle Loss ---
            # m1, m2: (N, D)
            # Normalize first
            m1 = F.normalize(m1, p=2, dim=1)
            m2 = F.normalize(m2, p=2, dim=1)
            
            # Compute confidence for downstream tasks (Heatmap/Coord loss) using Dual Softmax logic
            # Use a fixed temperature for confidence estimation
            temp = 10.0 # equivalent to 1/0.1
            sim_mat = torch.matmul(m1, m2.t()) * temp
            
            conf_matrix12 = F.log_softmax(sim_mat, dim=1)
            conf_matrix21 = F.log_softmax(sim_mat.t(), dim=1)
            with torch.no_grad():
                conf12 = torch.exp(conf_matrix12).max(dim=-1)[0]
                conf21 = torch.exp(conf_matrix21).max(dim=-1)[0]
                conf_per = conf12 * conf21
            
            # Compute Loss
            # Typically Circle Loss is calculated per anchor to handle positive/negative balance correctly
            # But here we have one-to-one matches (diagonal is positive, all others negative)
            # We can treat each row as an anchor.
            
            # Per-row loss calculation for better stability
            # N = num matches
            N = m1.size(0)
            
            # Positives: (N,)
            sp = torch.diag(raw_sim)
            
            # Negatives: (N, N-1) - for each anchor (row), other cols are negatives
            # We need to reshape the flat negatives back to (N, N-1) structure to compute per-sample loss
            mask = torch.eye(N, dtype=torch.bool, device=self.dev)
            sn = raw_sim[~mask].view(N, -1)
            
            # Vectorized Circle Loss (passing N anchors at once)
            loss_per = self.circle_loss(sp, sn)
            
             # --- Old Logic Commented Out ---
            # loss_per,conf_per=dual_softmax_loss(m1,m2)

            loss.append(loss_per.unsqueeze(0))
            conf_list.append(conf_per)
            
            acc_coarse_per=self.check_accuracy(m1,m2)
            acc += acc_coarse_per
            
        loss=torch.cat(loss,dim=-1).mean()
        acc /= B
        return loss,acc,conf_list
    
    
    def alike_distill_loss(self,kpts,alike_kpts):
        """
        ALike Keypoint Distillation Loss (ALike 知识蒸馏损失)
        
        功能:
            使用预训练的 ALike 模型提取的关键点作为伪标签（Pseudo-GT），通过知识蒸馏的方式监督
            当前模型的关键点提取器（Keypoint Extractor）。
            这使得模型不需要手工标注关键点（Keypoint Labels），而是直接学习优秀的特征响应图。
            
        计算:
            1. 构建 Label Map (H, W)，并根据 ALike 的关键点位置填充 Index。
            2. 使用 CrossEntropy（NLLLoss）来最小化预测关键点概率与 Label Map 的差异。
            3. 特殊采样：对背景区域（Negative）进行随机采样，避免类别不平衡（样本极度不均衡）。
        """
        C, H, W = kpts.shape
        kpts = kpts.permute(1,2,0)
        # get ALike keypoints
        with torch.no_grad():
            labels = torch.ones((H, W), dtype = torch.long, device = kpts.device) * 64 # -> Default is non-keypoint (bin 64)
            offsets = (((alike_kpts/8) - (alike_kpts/8).long())*8).long()
            offsets =  offsets[:, 0] + 8*offsets[:, 1]  # Linear IDX
            labels[(alike_kpts[:,1]/8).long(), (alike_kpts[:,0]/8).long()] = offsets

        kpts = kpts.view(-1,C)
        labels = labels.view(-1)

        mask = labels < 64
        idxs_pos = mask.nonzero().flatten()
        idxs_neg = (~mask).nonzero().flatten()
        perm = torch.randperm(idxs_neg.size(0))[:len(idxs_pos)//32]
        idxs_neg = idxs_neg[perm]
        idxs = torch.cat([idxs_pos, idxs_neg])

        kpts = kpts[idxs]
        labels = labels[idxs]

        with torch.no_grad():
            predicted = kpts.max(dim=-1)[1]
            acc =  (labels == predicted)
            acc = acc.sum() / len(acc)

        kpts = F.log_softmax(kpts,dim=-1)
        loss = F.nll_loss(kpts, labels, reduction = 'mean')

        return loss, acc
    
    
    def compute_keypoints_loss(self,kpts1,kpts2,alike_kpts1,alike_kpts2):
        loss=[]
        acc=0
        B,_,H,W=kpts1.shape
        
        for b in range(B):
            loss_per1,acc_per1=self.alike_distill_loss(kpts1[b],alike_kpts1[b])
            loss_per2,acc_per2=self.alike_distill_loss(kpts2[b],alike_kpts2[b])
            loss_per=(loss_per1+loss_per2)
            acc_per=(acc_per1+acc_per2)/2
            loss.append(loss_per.unsqueeze(0))
            acc += acc_per
            
        loss=torch.cat(loss,dim=-1).mean()
        acc /= B
        return loss,acc
    
    
    def compute_heatmaps_loss(self,heatmaps1,heatmaps2,pts,conf_list):
        """
        Heatmap Loss (热力图辅助回归损失)
        
        功能:
            用于辅助关键点定位的回归。通过描述子匹配信度（Confidence）引导热力图（Heatmap）的学习。
            如果描述子匹配得好（Confidence 高），则对应的热力图也应该有较强的响应。
            
        计算:
            Loss = L1(H_pred(pt), Conf(pt))
            - H_pred: 模型预测的热力图强度
            - Conf: 描述子双重Softmax计算出的“置信度”
            这相当于一种 Self-Supervised 的一致性约束（Consistency Constraint）。
        """
        loss=[]
        B,_,H,W=heatmaps1.shape
        
        for b in range(B):
            pts1,pts2=pts[b][:,:2],pts[b][:,2:]
            h1=heatmaps1[b,0,pts1[:,1].long(),pts1[:,0].long()]
            h2=heatmaps2[b,0,pts2[:,1].long(),pts2[:,0].long()]
            
            conf=conf_list[b]
            loss_per1=F.l1_loss(h1,conf)
            loss_per2=F.l1_loss(h2,conf)
            loss_per=(loss_per1+loss_per2)
            loss.append(loss_per.unsqueeze(0))
            
        loss=torch.cat(loss,dim=-1).mean()
        return loss
    
    
    def normal_loss(self,normal,target_normal):
        """
        Normal Loss (法向量一致性损失)
        
        功能:
            最小化预测法向量（Prediction Normal）与 Depth-Anything-V2 提取的伪 GT 法向量（Pseudo-GT）
            之间的夹角（Angular Distance）。
            用于增强模型对场景平面的感知能力，确保其能预测出正确的表面法向。
            这一损失仅对 MegaDepth 的 Pseudo-GT 区域有效（Depth-Anything 高置信度区域）。
            
        公式:
            Dot = <N_pred, N_GT> / |N_pred|*|N_GT|
            Angle = acos(Dot)
            Loss = mean(Angle[valid_mask])
            
        参数:
            normal (3, H, W): 预测法向量
            target_normal (3, H, W): GT法向量
        """
        # import pdb;pdb.set_trace()
        # Resize target_normal to match normal's spatial dimensions
        if normal.shape[1:] != target_normal.shape[1:]:
            target_normal = F.interpolate(target_normal.unsqueeze(0), size=normal.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

        normal = normal.permute(1, 2, 0)
        target_normal = target_normal.permute(1,2,0)
        # loss = F.l1_loss(d_feat, depth_anything_normal_feat)
        dot = torch.cosine_similarity(normal, target_normal, dim=2)
        valid_mask = target_normal[:, :, 0].float() \
                    * (dot.detach() < 0.999).float() \
                    * (dot.detach() > -0.999).float()
        valid_mask = valid_mask > 0.0
        al = torch.acos(dot[valid_mask])
        loss = torch.mean(al)
        return loss
    
    
    def compute_normals_loss(self,normals1,normals2,DA_normals1,DA_normals2,megadepth_batch_size,coco_batch_size):
        loss=[]
        
        # LiftFeat style: only MegaDepth image need depth-normal
        # Slice by skipping COCO images at the beginning
        if normals1.shape[0] > coco_batch_size:
            cur_normals1 = normals1[coco_batch_size:, ...]
            cur_normals2 = normals2[coco_batch_size:, ...]
        else:
            # Fallback if batch size is smaller than expected (e.g. dry run or config change)
            cur_normals1 = normals1
            cur_normals2 = normals2

        # Iterate over available Pseudo-GT normals
        for b in range(len(DA_normals1)):
            # Safety check to avoid index out of bounds if predictions are fewer than GT
            if b >= cur_normals1.shape[0]:
                break
                
            normal1, normal2 = cur_normals1[b], cur_normals2[b]
            loss_per1 = self.normal_loss(normal1, DA_normals1[b].permute(2, 0, 1))
            loss_per2 = self.normal_loss(normal2, DA_normals2[b].permute(2, 0, 1))
            loss_per = (loss_per1 + loss_per2)
            loss.append(loss_per.unsqueeze(0))
        
        if len(loss) == 0:
            return torch.tensor(0.0, device=self.dev, requires_grad=True)

        loss=torch.cat(loss,dim=-1).mean()
        return loss

    def _split_geo_features(self, geo_map):
        geo_dict = {}
        geo_idx = 0
        if self.geo_config.get('depth', False):
            geo_dict['depth'] = geo_map[:, geo_idx:geo_idx+1, :, :]
            geo_idx += 1
        if self.geo_config.get('normal', False):
            geo_dict['normal'] = geo_map[:, geo_idx:geo_idx+3, :, :]
            geo_idx += 3
        if self.geo_config.get('gradients', False):
            geo_dict['gradients'] = geo_map[:, geo_idx:geo_idx+2, :, :]
            geo_idx += 2
        if self.geo_config.get('curvatures', False):
            geo_dict['curvatures'] = geo_map[:, geo_idx:geo_idx+5, :, :]
            geo_idx += 5
        return geo_dict

    def compute_geometric_loss(self, geo_features1, geo_features2, DA_depths1, DA_depths2, megadepth_batch_size):
        """
        Geometric Loss (几何一致性损失：深度、梯度、曲率)
        
        功能:
            计算所有（可选的）几何特征的损失。这些几何特征不仅包括传统的深度（Depth），
            还包括通过 GeometricFeatureExtractor 计算得到的 derived 特征，如梯度（Gradient）
            和曲率（Curvature）。
            这些特征对大视差匹配极为有用，因为它们比 RGB 更有不变性。
            
        包含:
            - L_Depth: L1(MinMaxNorm(D_pred), MinMaxNorm(D_GT)) 最小化归一化深度的误差。
            - L_Grad: L1(Grad(D_pred), Grad(D_GT)) 最小化深度的空间梯度差异。
            - L_Curv: L1(Curv(D_pred), Curv(D_GT)) 最小化曲率（Mean/Gaussian/Principal Curvatures）差异。
            
        注意:
            只在 MegaDepth 数据（带深度信息）上计算，且需进行 Min-Max 归一化以消除绝对尺度带来的歧义（Scale Ambiguity）。
        """
        if isinstance(geo_features1, torch.Tensor):
            geo_features1 = self._split_geo_features(geo_features1)
        if isinstance(geo_features2, torch.Tensor):
            geo_features2 = self._split_geo_features(geo_features2)

        loss_depth_list = []
        loss_grad_list = []
        loss_curv_list = []
        
        # Only compute geometric loss for MegaDepth images
        if megadepth_batch_size > 0:
            # Helper to slice dict
            def slice_geo_dict(geo_dict, batch_size):
                sliced = {}
                for k, v in geo_dict.items():
                    sliced[k] = v[-batch_size:]
                return sliced

            cur_geo1 = slice_geo_dict(geo_features1, megadepth_batch_size)
            cur_geo2 = slice_geo_dict(geo_features2, megadepth_batch_size)
            
            # Handle DA_depths (Pseudo-GT)
            if len(DA_depths1) > megadepth_batch_size:
                cur_DA_depths1 = DA_depths1[-megadepth_batch_size:]
                cur_DA_depths2 = DA_depths2[-megadepth_batch_size:]
            else:
                cur_DA_depths1 = DA_depths1
                cur_DA_depths2 = DA_depths2
            
            # Check if 'depth' key exists, otherwise use length of any key
            sample_key = next(iter(cur_geo1))
            num_samples = min(len(cur_DA_depths1), cur_geo1[sample_key].shape[0])
            
            for b in range(num_samples):
                # Prepare target depth: ensure (1, 1, H, W)
                d_target1 = cur_DA_depths1[b]
                if d_target1.dim() == 2:
                    d_target1 = d_target1.unsqueeze(0).unsqueeze(0)
                elif d_target1.dim() == 3:
                    d_target1 = d_target1.unsqueeze(0)
                    
                d_target2 = cur_DA_depths2[b]
                if d_target2.dim() == 2:
                    d_target2 = d_target2.unsqueeze(0).unsqueeze(0)
                elif d_target2.dim() == 3:
                    d_target2 = d_target2.unsqueeze(0)
                
                # Get prediction size
                _, _, H_pred, W_pred = cur_geo1[sample_key].shape
                
                # Interpolate targets to match prediction size
                if d_target1.shape[-2:] != (H_pred, W_pred):
                    d_target1 = F.interpolate(d_target1, size=(H_pred, W_pred), mode='bilinear', align_corners=False)
                if d_target2.shape[-2:] != (H_pred, W_pred):
                    d_target2 = F.interpolate(d_target2, size=(H_pred, W_pred), mode='bilinear', align_corners=False)

                # Normalize target depth using Affine-Invariant Normalization (Min-Max)
                # This is robust to scale and shift ambiguity and guarantees [0, 1] range
                def min_max_norm(d):
                    d_min = d.min()
                    d_max = d.max()
                    if d_max - d_min > 1e-6:
                        return (d - d_min) / (d_max - d_min)
                    return torch.zeros_like(d)

                d_target1 = min_max_norm(d_target1)
                d_target2 = min_max_norm(d_target2)

                # Depth Loss
                if self.geo_config.get('depth', False) and 'depth' in cur_geo1:
                     d_pred1 = cur_geo1['depth'][b].unsqueeze(0)
                     d_pred2 = cur_geo2['depth'][b].unsqueeze(0)
                     
                     # Also normalize prediction for loss computation to ensure affine invariance
                     d_pred1_norm = min_max_norm(d_pred1)
                     d_pred2_norm = min_max_norm(d_pred2)
                     
                     l_d = self.lam_depth * (F.l1_loss(d_pred1_norm, d_target1) + F.l1_loss(d_pred2_norm, d_target2))
                     loss_depth_list.append(l_d.unsqueeze(0))

                # Compute derived features for TARGET if needed
                if self.geo_config.get('gradients', False) or self.geo_config.get('curvatures', False):
                    # Target features are computed from normalized depth
                    # Since d_target is [0, 1], GeometricExtractor won't divide by 255.
                    feat_target1 = self.curvature_computer(d_target1)
                    feat_target2 = self.curvature_computer(d_target2)

                    # Gradients
                    if self.geo_config.get('gradients', False):
                        # Extract predicted gradients (Channel 0: x, Channel 1: y)
                        pred_grad1 = cur_geo1['gradients'][b]
                        pred_grad2 = cur_geo2['gradients'][b]
                        
                        l_g = self.lam_gradients * (F.l1_loss(pred_grad1[0:1].unsqueeze(0), feat_target1['grad_x']) + \
                                    F.l1_loss(pred_grad1[1:2].unsqueeze(0), feat_target1['grad_y']) + \
                                    F.l1_loss(pred_grad2[0:1].unsqueeze(0), feat_target2['grad_x']) + \
                                    F.l1_loss(pred_grad2[1:2].unsqueeze(0), feat_target2['grad_y']))
                        loss_grad_list.append(l_g.unsqueeze(0))

                    # Curvatures
                    if self.geo_config.get('curvatures', False):
                        # Extract predicted curvatures
                        # Order: k1, k2, gaussian, mean, shape_index
                        pred_curv1 = cur_geo1['curvatures'][b]
                        pred_curv2 = cur_geo2['curvatures'][b]
                        
                        l_c = self.lam_curvature * (F.l1_loss(pred_curv1[0:1].unsqueeze(0), feat_target1['k1']) + \
                                    F.l1_loss(pred_curv1[1:2].unsqueeze(0), feat_target1['k2']) + \
                                    F.l1_loss(pred_curv1[2:3].unsqueeze(0), feat_target1['gaussian_curv']) + \
                                    F.l1_loss(pred_curv1[3:4].unsqueeze(0), feat_target1['mean_curv']) + \
                                    F.l1_loss(pred_curv1[4:5].unsqueeze(0), feat_target1['shape_index']) + \
                                    F.l1_loss(pred_curv2[0:1].unsqueeze(0), feat_target2['k1']) + \
                                    F.l1_loss(pred_curv2[1:2].unsqueeze(0), feat_target2['k2']) + \
                                    F.l1_loss(pred_curv2[2:3].unsqueeze(0), feat_target2['gaussian_curv']) + \
                                    F.l1_loss(pred_curv2[3:4].unsqueeze(0), feat_target2['mean_curv']) + \
                                    F.l1_loss(pred_curv2[4:5].unsqueeze(0), feat_target2['shape_index']))
                        loss_curv_list.append(l_c.unsqueeze(0))
        
        loss_depth = torch.cat(loss_depth_list).mean() if loss_depth_list else torch.tensor(0.0, device=self.dev)
        loss_grad = torch.cat(loss_grad_list).mean() if loss_grad_list else torch.tensor(0.0, device=self.dev)
        loss_curv = torch.cat(loss_curv_list).mean() if loss_curv_list else torch.tensor(0.0, device=self.dev)
            
        return loss_depth, loss_grad, loss_curv
    
    def coordinate_loss(self,coordinate,conf,pts1):
        """
        Coordinate Loss (坐标回归损失)
        
        功能:
            监督预测出的特征点坐标位置（Keypoint Location）。
            与通常的关键点检测（Heatmap）不同，此 loss 旨在让模型直接回归出精确的亚像素坐标。
            该 loss 根据匹配信度（Confidence）进行加权，即对于匹配得越好的点，其坐标越要准确。
            
        公式:
            Conf = Softmax(Conf_matrix)
            Loss = Conf * NLLLoss(LogSoftmax(Coord), Label)
            Label = Keypoint 8x8 Grid Index (Sub-pixel grid)
            
            相当于在一个 8x8 的 Grid 范围内预测离散化的 offset。
        """
        with torch.no_grad():
            coordinate_detached = pts1 * 8
            offset_detached = (coordinate_detached/8) - (coordinate_detached/8).long()
            offset_detached = (offset_detached * 8).long()
            label = offset_detached[:, 0] + 8*offset_detached[:, 1]

        #pdb.set_trace()
        coordinate_log = F.log_softmax(coordinate, dim=-1)

        predicted = coordinate.max(dim=-1)[1]
        acc =  (label == predicted)
        acc = acc[conf > 0.1]
        if len(acc) > 0:
            acc = acc.sum() / len(acc)
        else:
            acc = 0.0

        loss = F.nll_loss(coordinate_log, label, reduction = 'none')
        
        #Weight loss by confidence, giving more emphasis on reliable matches
        conf = conf / (conf.sum() + 1e-8)
        loss = (loss * conf).sum()

        return loss*2., acc
    
    def compute_deep_supervision_loss(self, preds, DA_depths, DA_normals, batch_size):
        """
        Deep Supervision Loss (深层监督损失)
        
        功能:
            对模型中间层的特征输出进行监督（Supervised Learning）。
            由于模型较深（Deep Network），早期层可能存在梯度消失或学习缓慢的问题。
            通过在中间层引入辅助 Loss，迫使中间特征也具有一定的几何语义（Partial Geometric Understanding），
            从而加速收敛并 regularize 模型。
        
        计算:
            Loss = L1(Downsampled(D_pred), Downsampled(D_GT)) + Cosine(Downsampled(N_pred), Downsampled(N_GT))
            
            这一过程需要将 GT 根据中间特征图的分辨率进行下采样（Downsampling）。
        """
        """
        Compute loss for intermediate predictions (pred_a, pred_b).
        preds: (B, 4, H, W) tensor. Channel 0 is depth, 1-3 is normal.
        DA_depths: List of GT depth maps (H_orig, W_orig)
        DA_normals: List of GT normal maps (H_orig, W_orig, 3)
        batch_size: Number of MegaDepth images to consider (at the end of the batch)
        """
        if batch_size <= 0:
            return torch.tensor(0.0, device=self.dev)

        loss_list = []
        
        # Slice predictions and GT
        cur_preds = preds[-batch_size:]
        
        if len(DA_depths) > batch_size:
            cur_DA_depths = DA_depths[-batch_size:]
            cur_DA_normals = DA_normals[-batch_size:]
        else:
            cur_DA_depths = DA_depths
            cur_DA_normals = DA_normals
            
        num_samples = min(len(cur_DA_depths), cur_preds.shape[0])
        _, _, H_pred, W_pred = cur_preds.shape
        
        for b in range(num_samples):
            # Prediction
            pred_d = cur_preds[b, 0:1] # (1, H, W)
            pred_n = cur_preds[b, 1:4] # (3, H, W)
            
            # Target Depth
            tgt_d = cur_DA_depths[b]
            if tgt_d.dim() == 2:
                tgt_d = tgt_d.unsqueeze(0).unsqueeze(0) # (1, 1, H_orig, W_orig)
            elif tgt_d.dim() == 3:
                tgt_d = tgt_d.unsqueeze(0) # (1, 1, H_orig, W_orig)
                
            # Target Normal
            tgt_n = cur_DA_normals[b] # (H_orig, W_orig, 3)
            tgt_n = tgt_n.permute(2, 0, 1).unsqueeze(0) # (1, 3, H_orig, W_orig)
            
            # Downsample Targets
            tgt_d_down = F.interpolate(tgt_d, size=(H_pred, W_pred), mode='bilinear', align_corners=False)
            tgt_n_down = F.interpolate(tgt_n, size=(H_pred, W_pred), mode='bilinear', align_corners=False)
            
            # Normalize Normals after interpolation
            tgt_n_down = F.normalize(tgt_n_down, p=2, dim=1)
            
            # Normalize Depths to 0-1 range using per-sample min-max
            d_min = tgt_d_down.min()
            d_max = tgt_d_down.max()
            if d_max - d_min > 1e-6:
                tgt_d_down = (tgt_d_down - d_min) / (d_max - d_min)
            else:
                tgt_d_down = torch.zeros_like(tgt_d_down)
            
            # Normalize Prediction Depth if needed (assuming prediction is raw output, but we want to match 0-1 target)
            # Note: pred_d is from LeakyReLU, so it's >= 0. 
            # If we assume the network learns to predict 0-1, we don't scale it.
            # But if we assume it predicts 0-255, we should scale it.
            # Given the main loss logic:
            # if d_pred1.max() > 1.0: d_pred1 = d_pred1 / 255.0
            # We should apply the same check.
            if pred_d.max() > 1.0:
                pred_d = pred_d / 255.0
                
            # Depth Loss (L1)
            l_d = F.l1_loss(pred_d, tgt_d_down.squeeze(0))
            
            # Normal Loss (Cosine Similarity)
            # Using the same logic as normal_loss function but adapted for downsampled tensors
            # normal_loss expects (3, H, W) inputs
            l_n = self.normal_loss(pred_n, tgt_n_down.squeeze(0))
            
            loss_list.append(l_d + l_n)
            
        if not loss_list:
            return torch.tensor(0.0, device=self.dev)
            
        return torch.stack(loss_list).mean()

    def compute_coordinates_loss(self,coordinates,pts,conf_list):
        loss=[]
        acc=0
        B,_,H,W=coordinates.shape
        
        for b in range(B):
            pts1,pts2=pts[b][:,:2],pts[b][:,2:]
            coordinate=coordinates[b,:,pts1[:,1].long(),pts1[:,0].long()].permute(1,0)
            conf=conf_list[b]
            
            loss_per,acc_per=self.coordinate_loss(coordinate,conf,pts1)
            loss.append(loss_per.unsqueeze(0))
            acc += acc_per
            
        loss=torch.cat(loss,dim=-1).mean()
        acc /= B
        
        return loss,acc
        
        
    def forward(self,
                descs1,fb_descs1,kpts1,normals1,
                descs2,fb_descs2,kpts2,normals2,
                geo_features1, geo_features2,
                pts,coordinates,fb_coordinates,
                alike_kpts1,alike_kpts2,
                DA_normals1,DA_normals2,
                DA_depths1, DA_depths2,
                megadepth_batch_size,coco_batch_size
                ):
        
        if isinstance(geo_features1, torch.Tensor):
            geo_features1 = self._split_geo_features(geo_features1)
        if isinstance(geo_features2, torch.Tensor):
            geo_features2 = self._split_geo_features(geo_features2)

        # import pdb;pdb.set_trace()
        self.loss_descs,self.acc_coarse,conf_list=self.compute_descriptors_loss(descs1,descs2,pts)
        self.loss_fb_descs,self.acc_fb_coarse,fb_conf_list=self.compute_descriptors_loss(fb_descs1,fb_descs2,pts)
        
        # start=time.perf_counter()
        self.loss_kpts,self.acc_kpt=self.compute_keypoints_loss(kpts1,kpts2,alike_kpts1,alike_kpts2)
        # end=time.perf_counter()
        # print(f"kpts loss cost {end-start} seconds")
        
        # Extract normals and depth from prediction (B, 4, H, W)
        # Channel 0: Depth, Channel 1-3: Normal
        if normals1.shape[1] == 4:
            pred_depths1 = normals1[:, 0:1, :, :]
            pred_normals1 = normals1[:, 1:4, :, :]
            pred_depths2 = normals2[:, 0:1, :, :]
            pred_normals2 = normals2[:, 1:4, :, :]
        else:
            # Fallback if model output changes
            pred_depths1 = None
            pred_normals1 = normals1
            pred_depths2 = None
            pred_normals2 = normals2

        # start=time.perf_counter()
        if self.geo_config.get('normal', True):
            self.loss_normals=self.compute_normals_loss(pred_normals1,pred_normals2,DA_normals1,DA_normals2,megadepth_batch_size,coco_batch_size)
        else:
            self.loss_normals = torch.tensor(0.0, device=self.dev)
        # self.loss_normals = self.compute_normals_loss(normals1, normals2, DA_normals1, DA_normals2,coco_batch_size)
        # end=time.perf_counter()
        # print(f"normal loss cost {end-start} seconds")
        
        if self.lam_depth > 0 or self.lam_gradients > 0 or self.lam_curvature > 0:
            self.loss_depths, self.loss_gradients, self.loss_curvature = self.compute_geometric_loss(geo_features1, geo_features2, DA_depths1, DA_depths2, megadepth_batch_size)
        else:
            self.loss_depths = torch.tensor(0.0, device=self.dev)
            self.loss_gradients = torch.tensor(0.0, device=self.dev)
            self.loss_curvature = torch.tensor(0.0, device=self.dev)

        # Deep Supervision Loss
        loss_ds_a = torch.tensor(0.0, device=self.dev)
        loss_ds_b = torch.tensor(0.0, device=self.dev)
        
        if self.lam_deep_supervision > 0 and megadepth_batch_size > 0:
            # Check for pred_a and pred_b in geo_features
            # We combine predictions from both images (1 and 2)
            
            # Image 1
            if 'pred_a' in geo_features1:
                loss_ds_a += self.compute_deep_supervision_loss(geo_features1['pred_a'], DA_depths1, DA_normals1, megadepth_batch_size)
            if 'pred_b' in geo_features1:
                loss_ds_b += self.compute_deep_supervision_loss(geo_features1['pred_b'], DA_depths1, DA_normals1, megadepth_batch_size)
                
            # Image 2
            if 'pred_a' in geo_features2:
                loss_ds_a += self.compute_deep_supervision_loss(geo_features2['pred_a'], DA_depths2, DA_normals2, megadepth_batch_size)
            if 'pred_b' in geo_features2:
                loss_ds_b += self.compute_deep_supervision_loss(geo_features2['pred_b'], DA_depths2, DA_normals2, megadepth_batch_size)
                
            # Average over images if both contributed
            if 'pred_a' in geo_features1 and 'pred_a' in geo_features2:
                loss_ds_a /= 2
            if 'pred_b' in geo_features1 and 'pred_b' in geo_features2:
                loss_ds_b /= 2
                
        self.loss_deep_supervision = loss_ds_a + loss_ds_b

        self.loss_coordinates,self.acc_coordinates=self.compute_coordinates_loss(coordinates,pts,conf_list)
        self.loss_fb_coordinates,self.acc_fb_coordinates=self.compute_coordinates_loss(fb_coordinates,pts,fb_conf_list)
        
        return {
            'loss_descs':self.lam_descs*self.loss_descs,
            'acc_coarse':self.acc_coarse,
            'loss_coordinates':self.lam_coordinates*self.loss_coordinates,
            'acc_coordinates':self.acc_coordinates,
            'loss_fb_descs':self.lam_fb_descs*self.loss_fb_descs,
            'acc_fb_coarse':self.acc_fb_coarse,
            'loss_fb_coordinates':self.lam_fb_coordinates*self.loss_fb_coordinates,
            'acc_fb_coordinates':self.acc_fb_coordinates,
            'loss_kpts':self.lam_kpts * self.loss_kpts,
            'acc_kpt':self.acc_kpt,
            'loss_normals':self.lam_normals * self.loss_normals,
            'loss_depths': self.lam_depth * self.loss_depths,
            'loss_gradients': self.lam_gradients * self.loss_gradients,
            'loss_curvature': self.lam_curvature * self.loss_curvature,
            'loss_deep_supervision': self.lam_deep_supervision * self.loss_deep_supervision
        }

