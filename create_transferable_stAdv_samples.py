# The code for the stAdv attack has been borrowed from: https://github.com/rakutentech/stAdv

import csv
from util import util
import metrics_and_dataloaders as mdl
import util.path as paths
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import optimize
import os
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

def flow_st(images,flows):
    images_shape = images.size()
    flows_shape = flows.size()
    batch_size = images_shape[0]
    H = images_shape[2]
    W = images_shape[3]
    basegrid = torch.stack(torch.meshgrid(torch.arange(0,H), torch.arange(0,W))) #(2,H,W)
    sampling_grid = basegrid.unsqueeze(0).type(torch.float32).cuda() + flows.cuda()
    sampling_grid_x = torch.clamp(sampling_grid[:,1],0.0,W-1.0).type(torch.float32)
    sampling_grid_y = torch.clamp(sampling_grid[:,0],0.0,H-1.0).type(torch.float32)

    x0 = torch.floor(sampling_grid_x).type(torch.int64)
    x1 = x0 + 1
    y0 = torch.floor(sampling_grid_y).type(torch.int64)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W - 2)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 2)
    y1 = torch.clamp(y1, 0, H - 1)

    Ia = images[:,:,y0[0,:,:], x0[0,:,:]]
    Ib = images[:,:,y1[0,:,:], x0[0,:,:]]
    Ic = images[:,:,y0[0,:,:], x1[0,:,:]]
    Id = images[:,:,y1[0,:,:], x1[0,:,:]]

    x0 = x0.type(torch.float32)
    x1 = x1.type(torch.float32)
    y0 = y0.type(torch.float32)
    y1 = y1.type(torch.float32)

    wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
    wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
    wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
    wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)
    
    perturbed_image = wa.unsqueeze(0)*Ia+wb.unsqueeze(0)*Ib+wc.unsqueeze(0)*Ic+wd.unsqueeze(0)*Id

    return perturbed_image.type(torch.float32).cuda()

def flow_loss(flows,padding_mode='constant', epsilon=1e-8):
    paddings = (1,1,1,1)
    padded_flows = F.pad(flows,paddings,mode=padding_mode,value=0)
    shifted_flows = [
    padded_flows[:, :, 2:, 2:],  # bottom right (+1,+1)
    padded_flows[:, :, 2:, :-2],  # bottom left (+1,-1)
    padded_flows[:, :, :-2, 2:],  # top right (-1,+1)
    padded_flows[:, :, :-2, :-2]  # top left (-1,-1)
    ]
    #||\Delta u^{(p)} - \Delta u^{(q)}||_2^2 + # ||\Delta v^{(p)} - \Delta v^{(q)}||_2^2 
    loss=0
    for shifted_flow in shifted_flows:
        loss += torch.sum(torch.square(flows[:, 1] - shifted_flow[:, 1]) + torch.square(flows[:, 0] - shifted_flow[:, 0]) + epsilon).cuda()
    return loss.type(torch.float32)

def metric_d(x0, x1, xref):
    d0 = metric.forward(xref,x0)
    d1 = metric.forward(xref,x1)
    return d0, d1

def rank_loss(s_adv, s_other):
    return (s_other/(s_adv+s_other)).float().clone()

def func(flows, x_prey, x_other, x_ref, alpha=50, beta=0.05):
    
    flows = torch.from_numpy(flows).view((1,2,)+x_prey.size()[2:]).cuda()
    flows.requires_grad=True
    pert_out = flow_st(x_prey,flows)
    
    s_adv, s_other = metric_d(pert_out, x_other, x_ref)
    
    L_adv = rank_loss(s_adv, s_other)
    
    L_flow = flow_loss(flows)
    
    L_final = alpha*L_adv+beta*L_flow
    
    metric.net.zero_grad()
    L_final.backward()
    
    gradient = flows.grad.data.view(-1).detach().cpu().numpy()
    
    if s_adv>s_other:
        return 0, gradient
    
    return L_final.item(), gradient

def attack(x_prey, x_other, x_ref, max_iter=50):
    init_flows = np.zeros((1,2,)+x_prey.size()[2:]).reshape(-1)
    results = optimize.fmin_l_bfgs_b(func, init_flows, args=(x_prey, x_other, x_ref), maxiter=max_iter, disp=False)
    return results

##### ##### ##### ##### #####

# Load metric
metric, transform, model_frwd_func = mdl.get_perceptual_similarity_metric("lpipsAlex", 64) # warning: hardcoded parameters for metric and bapps image size

# Setting GPU use to True here
use_gpu = True

# Load samples list
bapps_samples = list(np.load(paths.BAPPS_clearWinner_samples_file))

version = "YourDatasetVersionNameGoesHere"
results_path = "./datasets/transferableAdvSamples_"+version

if not os.path.isdir(results_path+"/stAdv/"):
    os.makedirs(results_path+"/stAdv/")

output_file = results_path+"/stAdv_flippedSamples.csv"
with open(output_file, mode='w') as f:
    w = csv.writer(f, delimiter=',')
    w.writerow(["path", "rank", "new_rank", "rmse", "d0", "d1", "dAdv"])
    
    for impath in tqdm(bapps_samples):
        
        x_0, x_1, x_ref = mdl.load_data(impath, transform)
        if use_gpu:
            x_0, x_1, x_ref = x_0.cuda(), x_1.cuda(), x_ref.cuda()

        d0, d1 = metric_d(x_0, x_1, x_ref)
        rank = int((d0>d1).item())

        if rank == 0:
            x_prey = x_0
            x_other = x_1
        else:
            x_prey = x_1
            x_other = x_0

        results = attack(x_prey, x_other, x_ref)
        
        flows = torch.from_numpy(results[0]).view((1,2,)+x_prey.size()[2:])
        
        x_adv = flow_st(x_prey, flows)
        
        s_adv = metric.forward(x_adv, x_ref)
        s_prey = metric.forward(x_prey, x_ref)
        s_other = metric.forward(x_other, x_ref)
        
        xprey_im = util.tensor2im(x_prey).astype(np.float32)
        xadv_im = util.tensor2im(x_adv).astype(np.float32)
        rmse = np.sqrt(np.square(xadv_im-xprey_im).mean())
        
        impathname = impath.split("/")
        
        impathname = results_path+"/stAdv/"+impathname[-3]+"_"+impathname[-1].split(".")[0]
        
        new_rank = rank
        if s_adv > s_other: # Saving image only if the rank flipped
            #rank flipped
            new_rank = int(not(rank))
            util.save_image(util.tensor2im(x_adv.detach()),impathname+".png")
        
        row = [impath, rank, new_rank, np.round(rmse,3)]
        if rank == 0:
            row += [np.round(s_prey.item(),3), np.round(s_other.item(),3), np.round(s_adv.item(),3)]
        else:
            row += [np.round(s_other.item(),3), np.round(s_prey.item(),3), np.round(s_adv.item(),3)]
        w.writerow(row)
        
        # print(s_adv.item(),s_prey.item(),s_other.item())
        # print(s_adv>s_other)
        # plt.imshow(util.tensor2im(x_prey))
        # plt.show()        
        # plt.imshow(util.tensor2im(pert_out))
        # plt.show()