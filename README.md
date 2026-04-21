# FESPL
Official pytorch implementation of our paper: [*Fourier-enhanced semi-supervised proxy learning for ultra-fine-grained novel class discovery*](https://www.sciencedirect.com/science/article/pii/S0031320325010301) (Pattern Recognition)


## Environment :snake:

Our implementation is based on [uno](https://github.com/DonkeyShot21/UNO), while logging is performed using `wandb`, we use `conda` to create the environment and install the dependencies.

Follow the commands below to setup environment.

```bash
# create environment
conda create -n rapl python=3.8
conda activate rapl

# choose the cudatoolkit version on your own
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch

# install other dependencies
pip install tqdm wandb scikit-learn pandas pytorch-lightning==1.1.3 lightning-bolts==0.3.0 

# create checkpoints directory
mkdir checkpoints
```

## Datasets :floppy_disk:

We use Ultra-FGVC as our main experiment datasets, specifically the `SoyAgeing-{R1, R3, R4, R5, R6}`.

You can download the datasets [here](https://github.com/XiaohanYu-GU/Ultra-FGVC), also make sure you have changed the datasets root that defined in [`config.py`](./config.py).

## Supervised Learning :star2:

```
CUDA_VISIBLE_DEVICES=0,1,2,4 python -m torch.distributed.launch --nproc_per_node 4  --master_port 12344 supervised_learning_aff.py \
           --dataset SoyAgeing-R1 \
           --task ncd --batch_size 8 \
           --model resnet50_aff \
           --image_size 448 \
           --weight_reg 1.0 \
           --weight_edl 1.0 \
           --num_proxy_base 4 \
           --old_classes_num 99
```

## Discovery Learning :sparkles:

```
CUDA_VISIBLE_DEVICES=0,1,2,4 python -m torch.distributed.launch --nproc_per_node 4  --master_port 12344 discovery_learning_aff.py \
          --dataset SoyAgeing-R1 \
          --task ncd --batch_size 8 \
          --model resnet50_aff \
          --image_size 448 \
          --pretrained ncd-resnet50_aff-SoyAgeing-R1-reg1.0-edl1.0-np4-ocn99.pth \
          --weight_contra 1.0 \
          --weight_edl 1.0 \
          --num_proxy_base 4 \
          --old_classes_num 99
```

## Citation :clipboard:

If you find our work helpful, please consider citing our paper:

```tex
@article{chen2025fourier,
  title={Fourier-Enhanced Semi-supervised Proxy Learning for Ultra-Fine-Grained Novel Class Discovery},
  author={Chen, Qiupu and Jiang, Hongkui and Jiao, Lin and Li, Zhou and Xu, Taosheng and Wang, Xue and Wang, Rujing},
  journal={Pattern Recognition},
  pages={112369},
  year={2026},
  publisher={Elsevier}
}
```

## Acknowledgements :gift:

- [UNO](https://github.com/DonkeyShot21/UNO)
- [Ultra-FGVC](https://github.com/XiaohanYu-GU/Ultra-FGVC)
- [UFG-NCD](https://github.com/SSDUT-Caiyq/UFG-NCD)
