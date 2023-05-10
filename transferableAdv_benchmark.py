import argparse
from tqdm import tqdm
import metrics_and_dataloaders as mdl
import csv
import util.path as paths

parser = argparse.ArgumentParser()
parser.add_argument("--save", type=str, default="", help="")
parser.add_argument("--metric", type=str, default="", help="")
conf = parser.parse_args()

def calc_metric_distance(x_ref, x_stAdvPGD, model_frwd_func=False):
    if model_frwd_func:
        distance = metric.forward(x_ref, x_stAdvPGD)
    else:
        distance = metric(x_ref, x_stAdvPGD)
    return distance

BAPPS_DATA_PATH = paths.BAPPS_dataset_path + "/val/"
ADV_IMGS_PATH = paths.Transferable_AdvSamples_path
ADV_IMGS_LIST_PATH = paths.BAPPS_stAdv_samples_list

output_file = "./results/transferableAdv_benchmark/" + conf.save + "_transferableAdv"

metric, transform, model_frwd_func = mdl.get_perceptual_similarity_metric(conf.metric, load_size=64)

ks = [5, 10, 15, 20]
with open(ADV_IMGS_LIST_PATH, "r") as f:
    stAdvPGDsamples = [s.strip() for s in f.readlines()]

with open(output_file + ".csv", mode="w") as f:
    w = csv.writer(f, delimiter=",")

    columns = ["sample", "d0", "d1", "dPGD10", "dPGD20", "dstAdv"]
    for k in ks:
        columns.append("dstAdvPGD" + str(k))
    w.writerow(columns)

    for sample in tqdm(stAdvPGDsamples):
        dist, img_id = sample.split("_")
        xref_path = BAPPS_DATA_PATH + dist + "/ref/" + img_id + ".png"
        xref = mdl.load_img(xref_path, transform)

        row = [sample]

        x0_path = BAPPS_DATA_PATH + dist + "/p0/" + img_id + ".png"
        x0 = mdl.load_img(x0_path, transform)
        d0 = calc_metric_distance(xref, x0, model_frwd_func)
        row.append(d0.item())

        x1_path = BAPPS_DATA_PATH + dist + "/p1/" + img_id + ".png"
        x1 = mdl.load_img(x1_path, transform)
        d1 = calc_metric_distance(xref, x1, model_frwd_func)
        row.append(d1.item())

        xPGD_path = ADV_IMGS_PATH + "PGD10/" + sample + ".png"
        xPGD = mdl.load_img(xPGD_path, transform)
        dPGD10 = calc_metric_distance(xref, xPGD, model_frwd_func)
        row.append(dPGD10.item())

        xPGD_path = ADV_IMGS_PATH + "PGD20/" + sample + ".png"
        xPGD = mdl.load_img(xPGD_path, transform)
        dPGD20 = calc_metric_distance(xref, xPGD, model_frwd_func)
        row.append(dPGD20.item())

        xstAdv_path = ADV_IMGS_PATH + "stAdv/" + sample + ".png"
        xstAdv = mdl.load_img(xstAdv_path, transform)
        dstAdv = calc_metric_distance(xref, xstAdv, model_frwd_func)
        row.append(dstAdv.item())

        for k in ks:
            stAdv_PGD_path = (
                ADV_IMGS_PATH + "stAdv_PGD" + str(k) + "/" + sample + ".png"
            )
            xstAdvPGD = mdl.load_img(stAdv_PGD_path, transform)
            dstAdvPGD = calc_metric_distance(xref, xstAdvPGD, model_frwd_func)
            row.append(dstAdvPGD.item())

        w.writerow(row)