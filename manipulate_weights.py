import torch
import numpy as np
import sys
import copy

##########################################
# For quantize-resistant trained networks
##########################################

fn = sys.argv[1]
tg = sys.argv[2]
iqp = int(sys.argv[3])
d = torch.load(fn, map_location=torch.device('cpu'))

qp = 1/iqp

nd = {}
ma = -100
mi = 100

bypass_key_list = ['latent_gen.h_analysis_2.kernel', 'latent_gen.h_analysis_2.b', 'latent_gen.h_analysis_2.kernel_init', 'latent_gen.h_analysis_2.b_init', 'latent_gen.gdn_2.beta', 'latent_gen.gdn_2.gamma', 'latent_gen.gdn_2.pedestal',]

key_list = ['entropy_coder.sigma', 'entropy_coder.mu', 'reconstructor.activation.beta', 'reconstructor.activation.gamma', 'reconstructor.activation.pedestal', 'reconstructor.up0.kernel', 'reconstructor.up0.b', 'reconstructor.conv0.kernel', 'reconstructor.conv0.b', 'reconstructor.up1.kernel', 'reconstructor.up1.b', 'reconstructor.conv1.kernel', 'reconstructor.conv1.b', 'reconstructor.up2.kernel', 'reconstructor.up2.b', 'reconstructor.conv2.kernel', 'reconstructor.conv2.b', 'reconstructor.conv2_cls.kernel', 'reconstructor.conv2_cls.b', 'reconstructor.likelihood_model.sigma', 'reconstructor.likelihood_model.mu']

handled_key_list = [
    'reconstructor.up0.kernel',
    'reconstructor.conv0.kernel',
    'reconstructor.up1.kernel',
    'reconstructor.conv1.kernel',
    'reconstructor.up2.kernel',
    'reconstructor.conv2.kernel',
    'reconstructor.conv2_cls.kernel']

BOUND = 1/4

with torch.no_grad():
    for k in bypass_key_list:
        nd[k] = d[k].clone()
    for k in key_list:
        if not k in handled_key_list:
            nd[k] = d[k].clone()
        else:
            w = d[k]
            nw = torch.round(w*iqp)
            # nw = torch.clamp(nw, -BOUND*iqp, BOUND*iqp)
            ma = max(ma, torch.max(nw).item())
            mi = min(mi, torch.min(nw).item())
            nw = nw / iqp
            nd[k] = nw
print(f'min: {mi}  max: {ma}')
torch.save(nd, tg)