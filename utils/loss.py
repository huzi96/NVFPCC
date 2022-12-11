from utils.misc import isin
import torch

def get_bce(data, groud_truth):
    """ Input data and ground_truth are sparse tensor.
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    mask = isin(data.C, groud_truth.C)
    bce = criterion(data.F.squeeze(), mask.type(data.F.dtype))
    bce /= torch.log(torch.tensor(2.0)).to(bce.device)
    sum_bce = bce * data.shape[0]
    
    return sum_bce

def get_focal(data, groud_truth, alpha=0.97, gamma=2):
    mask = isin(data.C, groud_truth.C)
    imask = ~mask
    F = data.F.squeeze() * ((-1)*imask + 1*mask)
    F = F + 1*imask
    alphas = alpha * ((-1)*imask + 1*mask)
    alphas = alphas + 1*imask
    losss = (-1) * alphas * ((1-F)**gamma) * torch.log(F)
    sum_loss = losss.sum()
    return sum_loss

def get_acc(data, groud_truth, thh=0.5, alpha=0.9, gamma=2):
    mask = isin(data.C, groud_truth.C)
    imask = ~mask
    tp = (data.F.squeeze() > thh) * mask
    tp = tp.sum()
    ap = mask.sum()

    tn = (data.F.squeeze() <= thh) * imask
    tn = tn.sum()
    an = imask.sum()
    return tp/ap, tn/an

def get_focal_legacy(data, groud_truth, out_sparse, alpha=0.97, gamma=2):
    mask = isin(out_sparse.detach().C, groud_truth.C)
    imask = ~mask
    F = data.reshape(-1) * ((-1)*imask + 1*mask)
    F = F + 1*imask
    alphas = alpha * ((-1)*imask + 1*mask)
    alphas = alphas + 1*imask
    losss = (-1) * alphas * ((1-F)**gamma) * torch.log(F)
    sum_loss = losss.sum()
    return sum_loss

def get_acc_legacy(data, groud_truth, out_sparse, thh=0.5, alpha=0.9, gamma=2):
    mask = isin(out_sparse.detach().C, groud_truth.C)
    imask = ~mask
    tp = (data.reshape(-1) > thh) * mask
    tp = tp.sum()
    ap = mask.sum()

    tn = (data.reshape(-1) <= thh) * imask
    tn = tn.sum()
    an = imask.sum()
    return tp/ap, tn/an

def get_focal_dense(data, groud_truth, alpha=0.97, gamma=2):
    assert data.shape == groud_truth.shape
    mask = groud_truth.bool()
    imask = ~mask 
    F = data * ((-1)*imask + 1*mask)
    F = F + 1*imask
    alphas = alpha * ((-1)*imask + 1*mask)
    alphas = alphas + 1*imask
    F = torch.clamp(F, min=1e-9)
    losss = (-1) * alphas * ((1-F)**gamma) * torch.log(F)
    sum_loss = losss.sum()
    return sum_loss

def get_acc_dense(data, groud_truth, thh=0.5, alpha=0.9, gamma=2):
    mask = groud_truth.bool()
    imask = ~mask
    tp = (data > thh) * mask
    tp = tp.sum()
    ap = mask.sum()

    tn = (data <= thh) * imask
    tn = tn.sum()
    an = imask.sum()
    return tp/ap, tn/an

def get_surface_loss_dense(data, groud_truth, dist, alpha=1):
    assert data.shape == groud_truth.shape
    assert dist.shape == groud_truth.shape
    posi_penalty = torch.square(dist) * data
    posi_gain = torch.pow(torch.square(dist)+1, -1) * data
    # sum_loss = posi_penalty.mean() - posi_gain.mean() * alpha
    return posi_penalty.mean(), posi_gain.mean()

def get_surf_focal_dense(data, groud_truth, dist, beta=1, alpha=0.97, gamma=2):
    assert data.shape == groud_truth.shape
    mask = groud_truth.bool()
    # try:
    #     assert (mask * dist).sum() < 1
    # except:
    #     import IPython
    #     IPython.embed()
    imask = ~mask
    dist_w = dist + mask * beta
    F = data * ((-1)*imask + 1*mask)
    F = F + 1*imask
    alphas = alpha * ((-1)*imask + 1*mask)
    alphas = alphas + 1*imask
    F = torch.clamp(F, min=1e-9)
    losss = (-1) * alphas * ((1-F)**gamma) * dist_w * torch.log(F)
    sum_loss = losss.sum()
    return sum_loss

def get_sse1(data, groud_truth, dist, thh, maxv=1023):
    pred = (data > thh).float()
    pred_dist = pred * dist
    sq_dist = torch.square(pred_dist)
    sse = sq_dist.sum()
    denom = pred.sum()
    # mse = sq_dist.mean()
    # psnr = 20*torch.log10(maxv / torch.sqrt(mse))
    return sse, denom

def get_se(data, dist, thh):
    pred = (data > thh).float()
    pred_dist = pred * dist
    sq_dist = torch.square(pred_dist)
    sq_dist = torch.cat([sq_dist, data], 1)
    return sq_dist

def get_surf_dual_dense(data, ground_truth, dist, beta=1):
    assert data.shape == ground_truth.shape
    mask = ground_truth.bool()
    loss_up = -torch.log(data+1e-6)*mask
    loss_up = loss_up.mean()
    pred_dist = data * torch.square(dist)
    sq_dist = pred_dist
    mse = sq_dist.mean()
    loss_down = mse
    loss = beta * loss_up + loss_down
    return loss, loss_up, loss_down