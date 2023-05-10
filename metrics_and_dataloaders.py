import torchvision.transforms as transforms
from PIL import Image
import sys
import util.path as paths

def transform_vNormalize(load_size=64):
    transform_list = []
    transform_list.append(transforms.Resize(load_size))
    transform_list += [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    return transform

def transform_vNoNormalization(load_size=64):
    transform_list = []
    transform_list.append(transforms.Resize(load_size))
    transform_list += [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    return transform

def load_data(im_path, transform):
    im_path = paths.BAPPS_dataset_path+im_path
    x0_path = im_path.replace("judge","p0").replace("npy","png")
    x1_path = im_path.replace("judge","p1").replace("npy","png")
    xref_path = im_path.replace("judge","ref").replace("npy","png")
    
    x0 = Image.open(x0_path).convert('RGB')
    x0 = transform(x0).unsqueeze(0)

    x1 = Image.open(x1_path).convert('RGB')
    x1 = transform(x1).unsqueeze(0)

    xref = Image.open(xref_path).convert('RGB')
    xref = transform(xref).unsqueeze(0)
    
    return x0, x1, xref

def load_img(im_path, transform):
    im = Image.open(im_path).convert('RGB')
    im = transform(im).unsqueeze(0)    
    return im.cuda()

def get_perceptual_similarity_metric(metric, load_size):
    if metric == "lpipsVgg" or metric == "lpipsSqueeze" or metric == "lpipsAlex":
        sys.path.insert(0,'../PerceptualSimilarity') # update path to cloned lpips repo
        from lpips import lpips
        if metric == "lpipsVgg":
            backbone = "vgg"
        elif metric == "lpipsSqueeze":
            backbone = "squeeze"
        else:
            backbone = "alex"
        opt = {
            "model":"lpips",
            "net":backbone,
            "model_path":None,
            "use_gpu":True,
            "gpu_ids":[0],
            "from_scratch":False,
            "train_trunk":False,
        }
        model = lpips.Trainer()
        model.initialize(model=opt['model'],
                        net=opt['net'],
                        model_path=opt['model_path'],
                        use_gpu=opt['use_gpu'],
                        pnet_rand=opt["from_scratch"],
                        pnet_tune=opt["train_trunk"],
                        gpu_ids=opt["gpu_ids"])
        metric = model
        transform_ = transform_vNormalize(load_size=load_size)
        model_frwd_func = True

    elif  "stlpips" in metric:
        from stlpips_pytorch import stlpips # pip install stlpips_pytorch
        if metric == "stlpipsAlex":
            print("Loading stlpipsAlex  .....")
            metric = stlpips.LPIPS(net="alex", variant="shift_tolerant").cuda()
        elif metric == "stlpipsVgg":
            print("Loading stlpipsVgg .....")
            metric = stlpips.LPIPS(net="vgg", variant="shift_tolerant").cuda()
        transform_ = transform_vNormalize(load_size=load_size)
        model_frwd_func = False
        
    elif metric == "l2":
        import torch
        def calc_l2(ref, dis):
            (N,C,X,Y) = ref.size()
            return torch.mean(torch.mean(torch.mean((ref-dis)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
        metric = calc_l2
        transform_ = transform_vNormalize(load_size=load_size)
        model_frwd_func = False

    elif metric == "ssim":
        sys.path.insert(0,'../pytorch-ssim') # https://github.com/Po-Hsun-Su/pytorch-ssim
        import pytorch_ssim
        ssim_loss = pytorch_ssim.SSIM()
        def calc_ssim(ref,dis):
            return (1. - ssim_loss(ref, dis)) / 2.
        metric = calc_ssim
        transform_ = transform_vNormalize(load_size=load_size)
        model_frwd_func = False

    elif metric == "msssim":
        from IQA_pytorch import MS_SSIM
        D = MS_SSIM(channels=3)
        def calc_msssim(ref,dis):
            return (1. - D(ref, dis, as_loss=False)) / 2.
        metric = calc_msssim
        transform_ = transform_vNormalize(load_size=load_size)

    # elif metric == "add new metric here"

    else:
        raise ValueError("Not a valid metric name.")

    return metric, transform_, model_frwd_func