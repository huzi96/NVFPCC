import numpy as np
import torch
import torch.nn as nn
import argparse

from torch import nn, optim
from torch.nn import parameter
import torch.nn.functional as F
import open3d as o3d
import time
import os
from torch.nn.modules.linear import Linear
import tqdm
import MinkowskiEngine as ME

from utils.dataloader import LoadedVoxelDataset
from utils.network import SingleLayerLatentGen, QuantGaussianLikelihood, CompDecoder, SingleLayerLatentGen
from utils.loss import get_focal_dense, get_acc_dense, get_surf_dual_dense, get_surf_focal_dense, get_sse1
import subprocess as sp
import util_code_quantized_weights as entropy_module

###########################################################################
param_model = 'Gaussian'
print(f'[Info] Using {param_model} for network parameters.')
prob_model = 'Gaussian'
print(f'[Info] Using {prob_model} for latent repr.')
main_loss = 'wfocal' # namely distance [w]eighted [focal] loss
focal_alpha = 0.9
print(f'[Info] Using {main_loss} as main loss function, alpha={focal_alpha}')
###########################################################################

class Net(nn.Module):
    def __init__(self, args, param_model, ch=4, channel_str='8,16,8,8') -> None:
        super(Net, self).__init__()
        print(f'[Net] Building model with latent channel {ch} and channel string {channel_str}')
        channels = tuple(np.array(channel_str.split(','), dtype=int))
        self.latent_gen = SingleLayerLatentGen(in_channels=ch, out_channels=ch)
        self.entropy_coder = QuantGaussianLikelihood(in_channels=ch)
        self.reconstructor = CompDecoder(args, param_model, useIGDN=True, in_channels=ch, channels=channels)
    
    def forward(self, emb, mode, q):
        latent = self.latent_gen(emb)
        latent_rounded, latent_likelihood = self.entropy_coder(latent, mode)
        out, out_cls_list, net_bits = self.reconstructor(latent_rounded, q)
        return out, out_cls_list, net_bits, latent_likelihood
    
    def reconstruct(self, latent, q):
        out, out_cls_list, net_bits = self.reconstructor(latent, q)
        return out
    
    def get_network_bits(self):
        nbits = self.entropy_coder.get_bits() + self.reconstructor.get_bits()
        return nbits
    
    def get_latent_bits(self, all_emb):
        latent = self.latent_gen(all_emb)
        _, latent_likelihood = self.entropy_coder(latent, mode='eval')
        return latent_likelihood.sum()

    def get_latent_code(self, all_emb):
        latent = self.latent_gen(all_emb)
        quantized_latent, latent_likelihood = self.entropy_coder(
            latent, mode='eval')
        sigma = torch.abs(self.entropy_coder.sigma)
        mu = self.entropy_coder.mu
        return {
            'quantized_latent': quantized_latent,
            'sigma': sigma,
            'mu': mu,
            'latent_likelihood': latent_likelihood
        }
    
    def get_bits(self, all_emb):
        return self.get_latent_bits(all_emb), self.get_network_bits()

class MultiscaleProcessor(nn.Module):
    def __init__(self):
        super(MultiscaleProcessor, self).__init__()
        self.down_sampler = nn.MaxPool3d(2, 2)
    
    def forward(self, x):
        x1 = self.down_sampler(x)
        x2 = self.down_sampler(x1)
        
        y_list = [x2, x1, x] 
        ground_truth_list = y_list 

        return ground_truth_list
    
class MultiscaleCounter(nn.Module):
    def __init__(self, niter=4):
        super(MultiscaleCounter, self).__init__()
        self.down_sampler = nn.AvgPool3d(2, 2)
        self.niter = niter
    
    def forward(self, x):
        with torch.no_grad():
            for i in range(self.niter):
                x = self.down_sampler(x)
                x = x * 8
        return x

def train():
    print(f'Rate loss = {args.w1} * b1 + b2 + {args.w2} * b3')
    device = torch.device('cuda')
    print('Buiding dataloader')
    fid = args.input[:-4]
    train_data = LoadedVoxelDataset(f'{fid}_l5_origins.npy', f'{fid}_l5_gt_grid.npy', f'{fid}_l5_dist.npy')
    training_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsize, shuffle=args.shuffle, num_workers=1, drop_last=False
    )
    print('Dataloader built')
    print('Using lambda: ', args.lmbda)
    ch=args.ch
    net = Net(args, param_model, ch, channel_str=args.chanstr).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)
    sch = optim.lr_scheduler.MultiStepLR(opt, [300, 400, 450], 0.1)
    ms_processor = MultiscaleProcessor().to(device)

    emb = torch.ones(
        (train_data.N_leaf,ch,2,2,2), dtype=torch.float32
        ).to(device)
    emb = emb.requires_grad_(True)
    opt_emb = optim.Adam([emb], lr=args.lr*args.wemb)
    print('Embedding learning rate: %f x %f = %f' % (args.lr, args.wemb, args.lr*args.wemb))
    sch_emb = optim.lr_scheduler.MultiStepLR(opt, [300, 400, 450], 0.1)

    for epoch in range(0, 501):
        start_time = time.time()
        cnt = 0
        list_train_loss = 0.
        list_pacc = []
        list_nacc = []
        list_msbces = []
        list_msacc = []
        list_bpp = []
        list_b_latent = []
        list_b_net = []
        list_posi_penal = []
        list_posi_gain = []
        list_sse = []
        list_denom = []

        if epoch == 0:
            q = 1
        if epoch == args.phase_change:
            q = 2

        for i, data in enumerate(training_loader, 0):
            opt.zero_grad()
            indices, x_dense, dist = data
            x_dense = x_dense.to(device)
            dist = dist.to(device)
            n_pts = x_dense.sum()
            indices = indices.reshape(-1)
            ground_truth_list = ms_processor(x_dense)

            emb_node = emb[indices]

            out, out_cls_list, net_bits, latent_bits = net(emb_node, 'train', q)
            b_latent = latent_bits.sum() / n_pts
            b_net = net_bits.sum() / train_data.N
            bpp = b_latent + b_net
            bpp_loss = b_latent * args.w1 + b_net * args.w2

            ms_bces = [
                get_focal_dense(
                    out_cls_list[0], ground_truth_list[0], alpha=0.85),
                get_focal_dense(
                    out_cls_list[1], ground_truth_list[1], alpha=0.85)
                ]
            ms_acc = [
                get_acc_dense(out_cls_list[0], ground_truth_list[0], thh=0.5),
                get_acc_dense(out_cls_list[1], ground_truth_list[1], thh=0.5)
            ]

            if main_loss == 'focal':
                bce = get_focal_dense(out, x_dense, alpha=focal_alpha)
                loss_up = torch.tensor(0).to(device)
                loss_down = torch.tensor(0).to(device)
            elif main_loss == 'surface':
                bce, loss_up, loss_down = get_surf_dual_dense(out, x_dense, dist, beta=0.5)
            elif main_loss == 'wfocal':
                bce = get_surf_focal_dense(out, x_dense, dist, beta=1, alpha=focal_alpha)
                loss_up = torch.tensor(0).to(device)
                loss_down = torch.tensor(0).to(device)
            else:
                raise NotImplementedError
            
            list_posi_penal.append(loss_down.cpu().item())
            list_posi_gain.append(loss_up.cpu().item())
            sse, denom = get_sse1(out, x_dense, dist, 0.6)
            list_sse.append(sse.item())
            list_denom.append(denom.item())

            loss = bce + ms_bces[0] + ms_bces[1] + args.lmbda * bpp_loss
            loss.backward()
            
            if torch.any(torch.isnan(loss)):
                print('Problem in loss')
                import IPython
                IPython.embed()
                raise ValueError()

            for p in net.parameters():
                if p.grad is None:
                    continue
                if torch.any(torch.isnan(p.grad)):
                    print('Problem with grad')
                    import IPython
                    IPython.embed()
                    raise ValueError()
            list_train_loss += loss.cpu().item()
            pacc, nacc = get_acc_dense(out, x_dense)
            list_pacc.append(pacc.item())
            list_nacc.append(nacc.item())
            list_msbces.append([i.item() for i in ms_bces])
            list_msacc.append([[i[0].item(), i[1].item()] for i in ms_acc])
            list_bpp.append(bpp.item())
            list_b_latent.append(b_latent.item())
            list_b_net.append(b_net.item())
            opt.step()
            cnt += 1

        ######## Update emb ####################################################
        opt_emb.zero_grad()
        x_dense_all, dist_all = train_data.get_all()
        x_dense_all = x_dense_all.to(device)
        dist_all = dist_all.to(device)
        n_pts = x_dense_all.sum()
        emb_all = emb

        ground_truth_list = ms_processor(x_dense_all)
        out, out_cls_list, net_bits, latent_bits = net(emb_all, 'train', q)
        b_latent = latent_bits.sum() / n_pts
        b_net = net_bits.sum() / train_data.N
        bpp = b_latent + b_net
        bpp_loss = b_latent * args.w1 + b_net * args.w2

        ms_bces = [
            get_focal_dense(
                out_cls_list[0], ground_truth_list[0], alpha=0.85),
            get_focal_dense(
                out_cls_list[1], ground_truth_list[1], alpha=0.85)
            ]
        
        bce = get_surf_focal_dense(out, x_dense_all, dist_all, beta=1, alpha=focal_alpha)
        loss = bce + ms_bces[0] + ms_bces[1] + args.lmbda * bpp_loss
        loss.backward()
        opt_emb.step()
        ####### Done ###########################################################

        sch_emb.step()
        sch.step()
        
        timestamp = time.time()
        d_list_msbces = np.array(list_msbces)
        d_list_msacc = np.array(list_msacc)
        mse1 = np.sum(list_sse) / np.sum(list_denom)
        psnr1 = 20*np.log10(1023/np.sqrt(mse1/3))
        print('[Epoch %04d TRAIN %.1f seconds] Loss: %.4e PosiPenal: %.4f PosiGain: %.4f Pacc: %.4f Nacc: %.4f S1 Loss: %.4f S2 Loss: %.4f S1Pacc: %.4f S1Nacc: %.4f S2Pacc: %.4f S2Nacc: %.4f bpp: %.4f b_latent: %.4f  b_net: %.4f MSE1: %.4f PSNR1: %.4f' % (
            epoch,
            timestamp - start_time,
            list_train_loss / cnt,
            np.sum(list_posi_penal) / cnt,
            np.sum(list_posi_gain) / cnt,
            np.sum(list_pacc) / cnt,
            np.sum(list_nacc) / cnt,
            np.sum(d_list_msbces[:,0]) / cnt,
            np.sum(d_list_msbces[:,1]) / cnt,
            np.sum(d_list_msacc[:,0,0]) / cnt,
            np.sum(d_list_msacc[:,0,1]) / cnt,
            np.sum(d_list_msacc[:,1,0]) / cnt,
            np.sum(d_list_msacc[:,1,1]) / cnt,
            np.sum(list_bpp) / cnt,
            np.sum(list_b_latent) / cnt,
            np.sum(list_b_net) / cnt,
            mse1,
            psnr1
        )
        )
        start_time = time.time()

        if epoch % 10 == 0:
            print('[INFO] Saving')
            if not os.path.isdir(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
            torch.save(net.state_dict(), './%s/%04d.ckpt' %
                       (args.checkpoint_dir, epoch))
            torch.save(emb, './%s/%04d_emb.ckpt' %
                       (args.checkpoint_dir, epoch))
            tq = 2

            cnt = 0
            list_train_loss = 0.
            list_pacc = []
            list_nacc = []
            list_msbces = []
            list_msacc = []
            list_bpp = []
            list_b_latent = []
            list_b_net = []
            list_posi_penal = []
            list_posi_gain = []
            list_sse = []
            list_denom = []

            with torch.no_grad():
                x_dense_all, dist_all = train_data.get_all()
                x_dense_all = x_dense_all.to(device)
                dist_all = dist_all.to(device)
                n_pts = x_dense_all.sum()
                emb_all = emb
                ground_truth_list = ms_processor(x_dense_all)

                out, out_cls_list, net_bits, latent_bits = net(emb_all, 'eval', tq)
                b_latent = latent_bits.sum() / n_pts
                b_net = net_bits.sum() / train_data.N
                assert n_pts.item() == train_data.N
                bpp = b_latent + b_net

                ms_bces = [
                    get_focal_dense(
                        out_cls_list[0], ground_truth_list[0], alpha=0.85),
                    get_focal_dense(
                        out_cls_list[1], ground_truth_list[1], alpha=0.85)
                    ]
                ms_acc = [
                    get_acc_dense(out_cls_list[0], ground_truth_list[0], thh=0.5),
                    get_acc_dense(out_cls_list[1], ground_truth_list[1], thh=0.5)
                ]
                if main_loss == 'focal':        
                    bce = get_focal_dense(out, x_dense_all, alpha=focal_alpha)
                    loss_up = torch.tensor(0).to(device)
                    loss_down = torch.tensor(0).to(device)
                elif main_loss == 'surface':
                    bce, loss_up, loss_down = get_surf_dual_dense(out, x_dense_all, dist, beta=0.5)
                elif main_loss == 'wfocal':
                    bce = get_surf_focal_dense(out, x_dense_all, dist_all, beta=1, alpha=focal_alpha)
                    loss_up = torch.tensor(0).to(device)
                    loss_down = torch.tensor(0).to(device)
                else:
                    raise NotImplementedError

                latent_bits, net_bits = net.get_bits(emb_all)
                bpp_all = (latent_bits.item() + net_bits.item()) / train_data.N

                list_posi_penal.append(loss_down.cpu().item())
                list_posi_gain.append(loss_up.cpu().item())
                loss = bce + ms_bces[0] + ms_bces[1] + args.lmbda * bpp
                list_train_loss += loss.cpu().item()
                pacc, nacc = get_acc_dense(out, x_dense_all)
                list_pacc.append(pacc.item())
                list_nacc.append(nacc.item())
                list_msbces.append([i.item() for i in ms_bces])
                list_msacc.append([[i[0].item(), i[1].item()] for i in ms_acc])
                list_bpp.append(bpp.item())
                list_b_latent.append(b_latent.item())
                list_b_net.append(b_net.item())

                sse, denom = get_sse1(out, x_dense_all, dist_all, 0.6)
                list_sse.append(sse.item())
                list_denom.append(denom.item())
                cnt += 1

                timestamp = time.time()
                d_list_msbces = np.array(list_msbces)
                d_list_msacc = np.array(list_msacc)
                mse1 = np.sum(list_sse) / np.sum(list_denom)
                psnr1 = 20*np.log10(1023/np.sqrt(mse1/3))
                print('[Epoch %04d TEST %.1f seconds] Loss: %.4e PosiPenal: %.4f PosiGain: %.4f Pacc: %.4f Nacc: %.4f S1 Loss: %.4f S2 Loss: %.4f S1Pacc: %.4f S1Nacc: %.4f S2Pacc: %.4f S2Nacc: %.4f bpp: %.4f b_latent: %.4f b_net: %.4f b_all: %.4f MSE1: %.4f PSNR1: %.4f' % (
                    epoch,
                    timestamp - start_time,
                    list_train_loss / cnt,
                    np.sum(list_posi_penal) / cnt,
                    np.sum(list_posi_gain) / cnt,
                    np.sum(list_pacc) / cnt,
                    np.sum(list_nacc) / cnt,
                    np.sum(d_list_msbces[:,0]) / cnt,
                    np.sum(d_list_msbces[:,1]) / cnt,
                    np.sum(d_list_msacc[:,0,0]) / cnt,
                    np.sum(d_list_msacc[:,0,1]) / cnt,
                    np.sum(d_list_msacc[:,1,0]) / cnt,
                    np.sum(d_list_msacc[:,1,1]) / cnt,
                    np.sum(list_bpp) / cnt,
                    np.sum(list_b_latent) / cnt,
                    np.sum(list_b_net) / cnt,
                    bpp_all,
                    mse1,
                    psnr1
                )
                )
                start_time = time.time()

def encode():
    """Pack all that is needed for decoding into an object."""
    thh = args.thh
    device = torch.device('cuda')
    fid = args.input[:-4]
    train_data = LoadedVoxelDataset(f'{fid}_l5_origins.npy', f'{fid}_l5_gt_grid.npy', f'{fid}_l5_dist.npy', shuffle=False)
    training_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsize, shuffle=False, num_workers=0, drop_last=True
    )
    net = Net(args, param_model=param_model, ch=args.ch, channel_str=args.chanstr).to(device)
    rc_pts = []

    #######################################################################
    # 1. Network parameters ###############################################
    net_weight_pack = entropy_module.enc_dec_from_file(args.load_weights)
    net_bits = len(net_weight_pack['bit_stream']) * 8
    #######################################################################

    d = torch.load(args.load_weights, map_location=device)
    nd = {}
    for k in d:
        if not 'init_coords' in k:
            nd[k] = d[k]
    rt = net.load_state_dict(nd, strict=False)
    emb = torch.load(args.load_emb).to(device)
    
    pruning = ME.MinkowskiPruning().to(device)
    list_pacc = []
    list_nacc = []
    list_bits = []
    list_sse = []
    list_denom = []
    list_se = []

    cnt = 0
    to_sparse = ME.MinkowskiToSparseTensor().to(device)
    origins = []
    N = len(train_data)

    #######################################################################
    # 2. Cube origins #####################################################
    for i in tqdm.tqdm(range(N)):
        origin = train_data.origins[i]
        origins.append(origin)
    np_origins = np.array(origins, dtype=np.int16)
    #######################################################################

    #######################################################################
    # 3. Embeddings #######################################################
    latent_info = net.get_latent_code(emb.to(device))
    print('Estimated bit rate: ', latent_info['latent_likelihood'].sum())
    def arithmetic_enc(tensor, sigma, mu):
        offset = 512
        s = tensor.shape
        np_tensor = tensor.detach().cpu().numpy()
        d = np_tensor.astype(np.int16)
        assert np.sum(np.abs(d - np_tensor)) < 1e-6
        flat_coeff = d.reshape(-1) + offset
        flat_sigma = torch.tile(
            sigma, (s[0], 1, s[2], s[3], s[4])
            ).detach().cpu().numpy().astype(np.float32).reshape((-1))
        flat_mu = torch.tile(
            mu, (s[0], 1, s[2], s[3], s[4])
            ).detach().cpu().numpy().astype(np.float32).reshape((-1)) + offset
        EXE_ARITH = './module_arithmeticcoding'
        length = np.array([flat_coeff.shape[0]], dtype=np.int64)
        cmd_s = length.tobytes() + flat_coeff.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes()  
        r = sp.run([EXE_ARITH, 'e', '1', '1'], input=cmd_s, stdout=sp.PIPE)
        latent_byte_stream = r.stdout
        cmd_s = length.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes() + latent_byte_stream
        print('Latent code byte-stream length: ', len(latent_byte_stream))
        c = sp.run(['./module_arithmeticcoding', 'd', '1', '1'], input=cmd_s, stdout=sp.PIPE)
        dec_latent_coeffs = np.frombuffer(c.stdout, dtype=np.int16)
        dec_latent_coeffs = dec_latent_coeffs - offset
        dec_latent_coeffs = dec_latent_coeffs.astype(np.float32).reshape(s)
        assert np.sum(np.abs(dec_latent_coeffs - np_tensor)) < 1e-6
        return {
            'shape': s,
            'latent_byte_stream': latent_byte_stream,
            'sigma': sigma,
            'mu': mu,
            'length': length
        }
    latent_pack = arithmetic_enc(
        latent_info['quantized_latent'],
        latent_info['sigma'],
        latent_info['mu']
        )
    #######################################################################

    #######################################################################
    total_pack = {
        'net_weight_pack': net_weight_pack,
        'origins': np_origins,
        'latent_pack': latent_pack
    }
    import pickle
    with open(args.pack_fn, 'wb') as f:
        pickle.dump(total_pack, f)
    #######################################################################
    
    print('Start to reconstruct')
    q = 2
    from utils.loss import get_acc, get_se
    density_cubes = []
    latent_rounded_list = []
    with torch.no_grad():
        if False:
            pass
        else:
            for i,data in enumerate(training_loader):
                indices, gt_grid, dist = data
                gt_grid = gt_grid.to(device)
                dist = dist.to(device)
                indices = indices.to(device).long().reshape(-1)
                x = to_sparse(gt_grid)
                mask = (x.F[:,0] > thh).bool()
                x = pruning(x, mask)
                x_dense = gt_grid

                x_emb = emb[indices]
                out_all, _, _, _ = net(x_emb, mode='eval', q=q)
                out_dense = out_all
                density_cubes.append(out_dense.cpu().numpy())

                out_all = to_sparse(out_dense)

                pacc, nacc = get_acc(out_all, x, thh=thh)
                sse, denom = get_sse1(out_dense, x_dense, dist, thh)
                se = get_se(out_dense, dist, thh)
                list_se.append(se.cpu().numpy())
                list_sse.append(sse.item())
                list_denom.append(denom.item())
                list_pacc.append(pacc.item())
                list_nacc.append(nacc.item())
                cnt += 1

                mask = (out_all.F[:,0] > thh).bool()
                out = pruning(out_all, mask)
                for j in range(args.batchsize):
                    coords1 = out.C[:,1:].cpu().numpy()
                    origin = origins[i]
                    scale = 1
                    pts = coords1 * scale + origin
                    rc_pts.append(pts)
            rc_pts = np.concatenate(rc_pts, 0)

            latent_bits = len(latent_pack['latent_byte_stream']) * 8

            print('[Latent code] Gross bpp: %.4f' % (
                        (latent_bits + net_bits) / train_data.N
                    )
                )
    rcpcd = o3d.geometry.PointCloud()
    rcpcd.points = o3d.utility.Vector3dVector(rc_pts)
    rcpts = np.asarray(rcpcd.points)
    rcpts = np.round(rcpts).astype(np.float64)
    rcpcd.points = o3d.utility.Vector3dVector(rcpts)
    o3d.visualization.draw_geometries([rcpcd])
    o3d.io.write_point_cloud('rc_enc.ply', rcpcd, write_ascii=True)


def decode():
    """Decode from a pack."""
    thh = args.thh
    device = torch.device('cuda')
    pack_fn = args.input
    net = Net(args, param_model=param_model, ch=args.ch, channel_str=args.chanstr)
    rc_pts = []

    #######################################################################
    # 1. network weights ##################################################
    import pickle
    with open(pack_fn, 'rb') as f:
        total_pack = pickle.load(f)
    net_weight_pack = total_pack['net_weight_pack']
    dec_pool = entropy_module.entropy_decode(
        net_weight_pack['bit_stream'],
        net_weight_pack['inv_codebook'],
        net_weight_pack['element_length'],
        net_weight_pack['shape_list'])
    nd = {}
    for k, v in zip(net_weight_pack['keys_quantize'], dec_pool):
        nd[k] = torch.from_numpy(v).float() / args.qp
    for k, v in zip(
        net_weight_pack['keys_code_as_is'], net_weight_pack['as_is_pool']):
        nd[k] = torch.from_numpy(v).float()
    rt = net.load_state_dict(nd, strict=False)
    net = net.to(device)
    #######################################################################

    #######################################################################
    # 2. latent codes #####################################################
    latent_pack = total_pack['latent_pack']
    latent_byte_stream = latent_pack['latent_byte_stream']
    length = latent_pack['length']
    sigma = latent_pack['sigma']
    mu = latent_pack['mu']
    s = latent_pack['shape']
    offset = 512
    flat_sigma = torch.tile(
        sigma, (s[0], 1, s[2], s[3], s[4])
        ).detach().cpu().numpy().astype(np.float32).reshape((-1))
    flat_mu = torch.tile(
        mu, (s[0], 1, s[2], s[3], s[4])
        ).detach().cpu().numpy().astype(np.float32).reshape((-1)) + offset
    cmd_s = length.tobytes() + flat_mu.tobytes() + flat_sigma.tobytes() + latent_byte_stream
    print('Latent code byte-stream length: ', len(latent_byte_stream))
    c = sp.run(['./module_arithmeticcoding', 'd', '1', '1'], input=cmd_s, stdout=sp.PIPE)
    dec_latent_coeffs = np.frombuffer(c.stdout, dtype=np.int16)
    dec_latent_coeffs = dec_latent_coeffs - offset
    dec_latent_coeffs = dec_latent_coeffs.astype(np.float32).reshape(s)
    dec_latent_coeffs = torch.from_numpy(dec_latent_coeffs).to(device)
    ######################################################################
    
    pruning = ME.MinkowskiPruning().to(device)
    cnt = 0
    to_sparse = ME.MinkowskiToSparseTensor().to(device)
    N = args.N

    #######################################################################
    # 3. Cube origins #####################################################
    np_origins = total_pack['origins']
    #######################################################################
    
    print('Start to reconstruct')
    q = 2
    from utils.loss import get_acc, get_se
    density_cubes = []
    with torch.no_grad():
        for i in range(N):
            # indices, coords, feats, _ = data
            latent = dec_latent_coeffs[i:i+1].to(device)
            out_all = net.reconstruct(latent, q=q)
            out_dense = out_all
            density_cubes.append(out_dense.cpu().numpy())
            out_all = to_sparse(out_dense)
            mask = (out_all.F[:,0] > thh).bool()
            out = pruning(out_all, mask)
            coords1 = out.C[:,1:].cpu().numpy()
            origin = np_origins[i]
            scale = 1
            pts = coords1 * scale + origin
            rc_pts.append(pts)

    rc_pts = np.concatenate(rc_pts, 0)
          
    rcpcd = o3d.geometry.PointCloud()
    rcpcd.points = o3d.utility.Vector3dVector(rc_pts)
    # rcpcd.scale(train_data.scale, np.array([0,0,0], dtype=np.float64))
    rcpts = np.asarray(rcpcd.points)
    # rcpts += train_data.means
    rcpts = np.round(rcpts).astype(np.float64)
    rcpcd.points = o3d.utility.Vector3dVector(rcpts)
    o3d.visualization.draw_geometries([rcpcd])
    o3d.io.write_point_cloud('rc_dec.ply', rcpcd, write_ascii=True)
    import IPython
    IPython.embed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "command", choices=["train",  "encode", "decode"],
        help="What to do?")
    parser.add_argument(    
        "input", nargs="?",
        help="Input filename.")
    parser.add_argument(
        "--checkpoint_dir", default="train",
        help="Directory where to save/load model checkpoints.")
    parser.add_argument(
        "--batchsize", type=int, default=2,
        help="Batch size for training.")
    parser.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    parser.add_argument(
        "--load_weights", default="",
        help="Weights to load")
    parser.add_argument(
        "--load_extern", default="",
        help="Load external weights")
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--alpha", type=float, default=200, help="Alpha.")
    parser.add_argument(
        "--use_coords", type=bool, default=False, help="Use coords?")
    parser.add_argument(
        "--real", type=bool, default=False, help="Real compression?")
    parser.add_argument(
        "--dsep", type=bool, default=False, help="Use depth-separable conv?")
    parser.add_argument(
        "--stat_latent", type=bool, default=False, help="Use empirical statistics as the entropy model for latent repr?")
    parser.add_argument(
        "--stat_net", type=bool, default=False, help="Use empirical statistics as the entropy model for network parameters?")

    parser.add_argument(
        "--w1", type=float, default=1, dest="w1",
        help="W1 for rate-distortion tradeoff.")
    parser.add_argument(
        "--w2", type=float, default=1, dest="w2",
        help="W2 for rate-distortion tradeoff.")
    
    parser.add_argument(
        "--notes", type=str, default="Hello", dest="notes",
        help="Leave a note?"
    )
    parser.add_argument(
        "--load_meta", type=str, default="", dest="load_meta",
        help="Load a meta init"
    )
    parser.add_argument(
        '--shuffle', type=bool, default=False, dest="shuffle",
        help="Shuffle the dataset randomly?"
    )

    parser.add_argument(
        "--phase_change", type=int, default=100, dest="phase_change",
        help="Phase change epoch.")

    parser.add_argument(
        "--wemb", type=float, default=5, dest="wemb",
        help="Weight for emb lr.")
    
    parser.add_argument(
        '--ch', type=int, default=8, dest="ch",
        help="# channels in latent"
    )

    parser.add_argument(
        "--load_emb", type=str, default="", dest="load_emb",
        help="Load an emb"
    )

    parser.add_argument(
        "--chanstr", type=str, default="8,16,8,8", dest="chanstr",
        help="Control channels in the compnet"
    )
    parser.add_argument(
        "--thh", type=float, default=0.6, dest="thh",
        help="Threshold.")
    parser.add_argument(
        '--pack_fn', default='pack.pk', help='package filename.')
    parser.add_argument(
        '--N', default=917, help='Number of leaves nodes.'
    )
    parser.add_argument(
        '--qp', type=float, default=16, help='Quantization parameter used for net weights.'
    )

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == 'encode':
        encode()
    elif args.command == 'decode':
        decode()
