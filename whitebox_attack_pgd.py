import csv
import numpy as np
from util import util
import metrics_and_dataloaders as mdl
import torch
import copy
import argparse
from tqdm import tqdm
import util.path as paths

parser = argparse.ArgumentParser()
parser.add_argument("--save", type=str, default="", help="save results with the following name [lpipsAlex], default is empty string")
parser.add_argument("--load_size", type=int, default=64, help="Resize image to the following resolution [64]")
parser.add_argument("--metric", type=str, default="lpipsAlex", help="[lpipsAlex], [lpipsVgg], [lpipsSqueeze], [stlpipsAlex], [stlpipsVgg], [l2] or [ssim] for perceptual metric")
conf = parser.parse_args()

def pgd_attack(x_0, x_1, x_ref, model, model_frwd_func=True, eps=0.03, alpha=0.001, max_iters=30):
    if model_frwd_func:
        s0 = model.forward(x_ref, x_0)
        s1 = model.forward(x_ref, x_1)
    else:
        s0 = model(x_ref, x_0)
        s1 = model(x_ref, x_1)
    rank = int((s0 > s1).item())

    loss_func = torch.nn.MSELoss()

    if rank == 1:
        x_i = copy.deepcopy(x_1)
        x_orig = x_1
        s_other = s0
    else:
        x_i = copy.deepcopy(x_0)
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
    return x_i, s0, s1, s_adv, rank, new_rank, i

# Load samples list
bapps_samples = list(np.load(paths.BAPPS_clearWinner_samples_file))

# Create output file
output_file = "results/whitebox_attack/" + conf.save + "_pgd"

metric, transform, model_frwd_func = mdl.get_perceptual_similarity_metric(conf.metric, conf.load_size)

with open(output_file + ".csv", mode="w") as f:
    w = csv.writer(f, delimiter=",")
    w.writerow(["path", "new_h", "orig_h", "i", "rmse", "d0", "d1", "di", "0.001", "0.005", "0.01", "0.02", "0.03", "0.05", "0.1"])

    for impath in tqdm(bapps_samples):
        # Load sample
        x_0, x_1, x_ref = mdl.load_data(impath, transform)
        x_0, x_1, x_ref = x_0.cuda(), x_1.cuda(), x_ref.cuda()

        # Attack metric
        perturbed_data, d0, d1, di, original_rank, new_rank, i = pgd_attack(
            x_0, x_1, x_ref, metric, model_frwd_func
        )

        # Calculate RMSE
        pd = util.tensor2im(perturbed_data).astype(np.float32)
        if original_rank == 1:
            x1 = util.tensor2im(x_1).astype(np.float32)
            rmse = np.sqrt(np.square(x1 - pd).mean())
            x_diff = perturbed_data - x_1
        else:
            x0 = util.tensor2im(x_0).astype(np.float32)
            rmse = np.sqrt(np.square(x0 - pd).mean())
            x_diff = perturbed_data - x_0

        # Save in csv
        row = [
            impath,
            new_rank,
            original_rank,
            i,
            np.round(rmse, 3),
            np.round(d0.item(), 3),
            np.round(d1.item(), 3),
            np.round(di.item(), 3),
        ]
        for i in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
            row += [
                np.round(
                    (
                        torch.sum((torch.abs(x_diff.cpu()) > i).int()).float()
                        / (conf.load_size * conf.load_size * 3)
                    ).numpy(),
                    3,
                )
            ]
        w.writerow(row)
