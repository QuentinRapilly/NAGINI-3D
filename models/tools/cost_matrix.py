import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

K = 10
BINS = [k/K for k in range(K+1)]


def compute_cost_matrix(GT_mask, pred_mask):
    f_GT = GT_mask.flatten()
    f_pred = pred_mask.flatten()
    M_GT = f_GT.max()
    M_pred = f_pred.max()

    intersection = np.zeros((M_GT+1,M_pred+1))
    S_GT = np.zeros((M_GT+1))
    S_pred = np.zeros((M_pred+1))

    I = len(f_GT)

    for i in range(I):
        intersection[f_GT[i],f_pred[i]] += 1
        S_GT[f_GT[i]] +=1
        S_pred[f_pred[i]]+= 1


    union = S_GT[:,None] + S_pred[None,:] - intersection
    iou = intersection/union

    iou_no_bg = iou[1:,1:]

    return iou_no_bg


def compute_jaccard(GT_mask, pred_mask, bins=BINS):
    C = compute_cost_matrix(GT_mask, pred_mask)
    N_GT, N_pred = C.shape
    row_ind, col_ind = linear_sum_assignment(cost_matrix=C, maximize=True)
    iou = C[row_ind, col_ind]
    res, _ = np.histogram(iou, bins)
    TP = np.cumsum(res[::-1])[::-1]

    return TP, N_GT, N_pred


def iou_at_centers(GT_mask, pred_mask):

    cells_iou = list()
    idx_cells = torch.unique(GT_mask)[1:]

    for k,idx in enumerate(idx_cells):

        crt_GT = (GT_mask == idx)
        crt_pred = (pred_mask == idx)

        inter = (crt_GT*crt_pred).sum()
        union = crt_GT.sum() + crt_pred.sum() - inter

        IoU = inter/union

        cells_iou.append(IoU.item())

    return np.array(cells_iou)