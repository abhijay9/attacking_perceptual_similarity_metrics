# Attacking Perceptual Similarity Metrics

[Abhijay Ghildyal](https://abhijay9.github.io/), [Feng Liu](http://web.cecs.pdx.edu/~fliu/). In TMLR, 2023. [[OpenReview]](https://openreview.net/forum?id=r9vGSpbbRO) [[Arxiv]]()

<img src="imgs/teaser.png" width=400>

Figure (above): $I_1$ is more similar to $I_{ref}$ than $I_{0}$ according <br/> 
to all perceptual similarity metrics and humans. We attack <br/>
$I_1$ by adding imperceptible adversarial perturbations ($\delta$) <br/>
such that the metric ($f$) flips its earlier assigned rank, i.e., <br/>
in the above sample, $I_0$ becomes more similar to $I_{ref}$.

<br/>

<img src="https://abhijay9.github.io/images/lpips_pgd.gif" width=300 />
Figure (above): An example of the PGD attack on LPIPS(Alex)

## Requirements

Requires Python 3+ and PyTorch 0.4+. For evaluation, please download the data from the links below. 

When starting this project, I used the `requirements.txt` [(link)](https://github.com/richzhang/PerceptualSimilarity/blob/master/requirements.txt) from the LPIPS repository [(link)](https://github.com/richzhang/PerceptualSimilarity/). We are grateful to the authors of various perceptual similarity metrics for making their code and data publicly accessible.

## Downloads

The transferable adversarial attack samples generated for our benchmark in Table 5 can be downloaded from this [google drive folder (link)](https://drive.google.com/drive/folders/1uocGBWYrxAogMYlHaqFyidB-pjXVcJ7H?usp=sharing). Please unzip `transferableAdvSamples.zip` in the `datasets/` folder.

Alternatively, you can use the following:
```
cd datasets
gdown 1gA7lD7FtvssQoMQwaaGS_6E3vPkSf66T # get <id> from google drive (see below)
unzip transferableAdvSamples.zip
```
In case the gdown id changes, you can obtain it from the 'shareable with anyone' link for `transferableAdvSamples.zip` file in the aforementioned Google Drive folder. The id will be a substring in the shareable link, as shown here: `https://drive.google.com/file/d/<id>/view?usp=share_link`.

Download the LPIPS repo [(link)](https://github.com/richzhang/PerceptualSimilarity), outside this folder. Then, download the BAPPS dataset as mentioned here: [link](https://github.com/richzhang/PerceptualSimilarity#2-berkeley-adobe-perceptual-patch-similarity-bapps-dataset).

## Benchmark

Use the following to benchmark various metrics on the transferable adversarial samples created by attacking LPIPS(Alex) on BAPPS dataset samples via stAdv and PGD.

```
# L2
CUDA_VISIBLE_DEVICES=0 python transferableAdv_benchmark.py --metric l2 --save l2

# SSIM
CUDA_VISIBLE_DEVICES=0 python transferableAdv_benchmark.py --metric ssim --save ssim

# ST-LPIPS(Alex)
CUDA_VISIBLE_DEVICES=0 python transferableAdv_benchmark.py --metric stlpipsAlex --save stlpipsAlex
```

The results will be stored in the `results/transferableAdv_benchmark/` folder. 

Finally, use the ipython notebook `results/study_results_transferableAdv_attack.ipynb` to calculate the number of flips.

## Citation

If you find this repository useful for your research, please use the following to cite our work:

```
@article{ghildyal2023attackPercepMetrics,
  title={Attacking Perceptual Similarity Metrics},
  author={Abhijay Ghildyal and Feng Liu},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```
