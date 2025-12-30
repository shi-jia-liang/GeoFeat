import numpy as np
import torch
import poselib


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err

def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    M, info = poselib.estimate_relative_pose(
        kpts0, kpts1,
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
        {"max_epipolar_error": thresh,
         "success_prob": conf,
         "min_iterations": 20,
         "max_iterations": 1_000},
    )

    R, t, inl = M.R, M.t, info["inliers"]
    inl = np.array(inl)
    ret = (R, t, inl)

    return ret

def tensor2bgr(t):
    """将 tensor (C, H, W) 转换为 BGR 图像 (H, W, 3)"""
    # 处理可能是稀疏张量的情况
    if t.is_sparse:
        t = t.coalesce().to_dense()
    
    # 确保是密集张量
    if not isinstance(t, torch.Tensor) or t.is_sparse:
        t = torch.tensor(t)
    
    # 取第一个 batch（如果有）或直接使用
    if t.dim() == 4:
        t = t[0]  # 移除 batch 维度
    elif t.dim() == 2:
        # 如果是 (H, W) 格式，假设是灰度图
        t = t.unsqueeze(0).repeat(3, 1, 1)  # 转换为 (3, H, W)
    
    # 执行 permute 操作
    if t.dim() == 3:
        result = t.permute(1, 2, 0).cpu().numpy()
    else:
        result = t.cpu().numpy()
    
    return (result * 255).astype(np.uint8)

def compute_pose_error(match_fn,data):
    result = {}
    
    with torch.no_grad():
        mkpts0, mkpts1 = match_fn(tensor2bgr(data["image0"]), tensor2bgr(data["image1"]))

    # 获取 scale 因子（可能是 tensor 或 numpy）
    scale0 = data["scale0"]
    scale1 = data["scale1"]
    if isinstance(scale0, torch.Tensor):
        scale0 = scale0.numpy()
    if isinstance(scale1, torch.Tensor):
        scale1 = scale1.numpy()
    
    mkpts0 = mkpts0 * scale0
    mkpts1 = mkpts1 * scale1

    K0 = data["K0"].numpy() if isinstance(data["K0"], torch.Tensor) else data["K0"]
    K1 = data["K1"].numpy() if isinstance(data["K1"], torch.Tensor) else data["K1"]
    T_0to1 = data["T_0to1"].numpy() if isinstance(data["T_0to1"], torch.Tensor) else data["T_0to1"]

    # 处理形状问题
    if K0.ndim == 3:
        K0 = K0[0]
    if K1.ndim == 3:
        K1 = K1[0]
    if T_0to1.ndim == 3:
        T_0to1 = T_0to1[0]

    result = {}
    conf = 0.99999
    
    ret = estimate_pose(mkpts0, mkpts1, K0, K1, 4.0, conf)
    if ret is not None:
        R, t, inliers = ret
        t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
        result['R_err'] = R_err
        result['t_err'] = t_err

    return result


def error_auc(errors, thresholds=[5, 10, 20]):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []

    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def compute_maa(pairs, thresholds=[5, 10, 20]):
    # print("auc / mAcc on %d pairs" % (len(pairs)))
    errors = []

    for p in pairs:
        et = p['t_err']
        er = p['R_err']
        errors.append(max(et, er))

    d_err_auc = error_auc(errors)

    # for k,v in d_err_auc.items():
    #     print(k, ': ', '%.1f'%(v*100))

    errors = np.array(errors)

    for t in thresholds:
        acc = (errors <= t).sum() / len(errors)
        # print("mAcc@%d: %.1f "%(t, acc*100))
        
    return d_err_auc,errors
        
def homo_trans(coord, H):
    kpt_num = coord.shape[0]
    homo_coord = np.concatenate((coord, np.ones((kpt_num, 1))), axis=-1)
    proj_coord = np.matmul(H, homo_coord.T).T
    proj_coord = proj_coord / proj_coord[:, 2][..., None]
    proj_coord = proj_coord[:, 0:2]
    return proj_coord    
