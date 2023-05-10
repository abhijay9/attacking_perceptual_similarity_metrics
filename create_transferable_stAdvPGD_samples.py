from util import util
import metrics_and_dataloaders as mdl
import pandas as pd
import torch
from tqdm import tqdm
import os
import copy
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


def pgd_attack(x_0, x_1, x_ref, x_stAdv, model, model_frwd_func=True, eps=0.03, alpha=0.001, max_iters=20):
    if model_frwd_func:
        s0 = model.forward(x_ref, x_0)
        s1 = model.forward(x_ref, x_1)
    else:
        s0 = model(x_ref, x_0)
        s1 = model(x_ref, x_1)
    rank = int((s0 > s1).item())

    loss_func = torch.nn.MSELoss()

    x_i = x_stAdv

    if rank == 1:
        x_orig = x_1
        s_other = s0
    else:
        x_orig = x_0
        s_other = s1

    new_rank = rank
    for i in range(max_iters):
        x_i.requires_grad = True
        if model_frwd_func:
            s_adv = model.forward(x_ref, x_i)
        else:
            s_adv = model(x_ref, x_i)

        # If rank has changed then break
        if s_adv > s_other:
            # Attack successful
            new_rank = int(not (rank))
            x_i = x_i.detach_()
            break

        # Zero all existing gradients
        if model_frwd_func:
            model.net.zero_grad()
        else:    
            model.zero_grad()

        # Compute model score for adversarial image
        if model_frwd_func:
            s_adv = model.forward(x_ref, x_i)
        else:
            s_adv = model(x_ref, x_i)

        loss = loss_func(
            (s_other / (s_other + s_adv)).float(), torch.tensor([1]).float().cuda()
        )

        # Calculate gradients of model in backward pass
        loss.backward(retain_graph=True)

        adv_img = x_i + alpha * x_i.grad.sign()
        eta = torch.clamp(adv_img - x_orig, min=-eps, max=eps)
        x_i = torch.clamp(x_orig + eta, min=-1, max=1).detach_()

    # Return the perturbed image
    return x_i, s0, s1, s_adv, rank


##### ##### ##### ##### #####

# Load metric
metric, transform, model_frwd_func = mdl.get_perceptual_similarity_metric("lpipsAlex", 64) # warning: hardcoded parameters for metric and bapps image size

# Setting GPU use to True here
use_gpu = True

version = "YourDatasetVersionNameGoesHere"
results_path = "./datasets/transferableAdvSamples_"+version

# Load samples list
stadv_samples = pd.read_csv(results_path+"/stAdv_flippedSamples.csv")

max_pgd_attack_iters = 20

stadv_samples = stadv_samples[stadv_samples["rank"]!=stadv_samples["new_rank"]]

for impath in tqdm(list(stadv_samples["path"])):

    sample_id = impath.split("/")[-3]+"_"+impath.split("/")[-1].split(".npy")[0]

    x_0, x_1, x_ref = mdl.load_data(impath, transform)
    if use_gpu:
        x_0, x_1, x_ref = x_0.cuda(), x_1.cuda(), x_ref.cuda()
    
    x_stAdv_path =  results_path + "/stAdv/" + sample_id + ".png"
    x_stAdv = mdl.load_img(x_stAdv_path, transform)

    x_adv, d0, d1, dAdv, rank = pgd_attack(x_0, x_1, x_ref, x_stAdv, metric, model_frwd_func=True, max_iters=max_pgd_attack_iters)
    
    if not os.path.isdir(results_path+"/stAdv_PGD"+str(max_pgd_attack_iters)+"/"):
        os.makedirs(results_path+"/stAdv_PGD"+str(max_pgd_attack_iters)+"/")

    util.save_image(util.tensor2im(x_adv), results_path+"/stAdv_PGD"+str(max_pgd_attack_iters)+"/"+sample_id+".png")