# Big Vision

This codebase is designed for training large-scale vision models using
[Cloud TPU VMs](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms)
or GPU machines. It is based on [Jax](https://github.com/google/jax)/[Flax](https://github.com/google/flax)
libraries, and uses [tf.data](https://www.tensorflow.org/guide/data) and
[TensorFlow Datasets](https://www.tensorflow.org/datasets) for scalable and
reproducible input pipelines.

The open-sourcing of this codebase has two main purposes:
1. Publishing the code of research projects developed in this codebase (see a
   list below).
2. Providing a strong starting point for running large-scale vision experiments
   on GPU machines and Google Cloud TPUs, which should scale seamlessly and
   out-of-the box from a single TPU core to a distributed setup with up to 2048
   TPU cores.

`big_vision` aims to support research projects at Google. We are unlikely to
work on feature requests or accept external contributions, unless they were
pre-approved (ask in an issue first). For a well-supported transfer-only
codebase, see also [vision_transformer](https://github.com/google-research/vision_transformer).

Note that `big_vision` is quite dynamic codebase and, while we intend to keep
the core code fully-functional at all times, we can not guarantee timely updates
of the project-specific code that lives in the `.../proj/...` subfolders.
However, we provide a [table](#project-specific-commits) with last known
commits where specific projects were known to work.

The following research projects were originally conducted in the `big_vision`
codebase:

### Architecture research

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), by
  Alexey Dosovitskiy*, Lucas Beyer*, Alexander Kolesnikov*, Dirk Weissenborn*,
  Xiaohua Zhai*, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
  Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby*
- [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560), by
  Xiaohua Zhai*, Alexander Kolesnikov*, Neil Houlsby, and Lucas Beyer*\
  Resources: [config](big_vision/configs/proj/scaling_laws/train_vit_g.py).
- [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270), by
  Andreas Steiner*, Alexander Kolesnikov*, Xiaohua Zhai*, Ross Wightman,
  Jakob Uszkoreit, and Lucas Beyer*
- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601), by
  Ilya Tolstikhin*, Neil Houlsby*, Alexander Kolesnikov*, Lucas Beyer*,
  Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner,
  Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy\
  Resources: [config](big_vision/configs/mlp_mixer_i1k.py).
- [Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580), by
  Lucas Beyer, Xiaohua Zhai, Alexander Kolesnikov\
  Resources: [config](big_vision/configs/vit_s16_i1k.py)
- [UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes](https://arxiv.org/abs/2205.10337), by
  Alexander Kolesnikov^*, André Susano Pinto^*, Lucas Beyer*, Xiaohua Zhai*, Jeremiah Harmsen*, Neil Houlsby*\
  Resources: [readme](big_vision/configs/proj/uvim/README.md), [configs](big_vision/configs/proj/uvim), [colabs](big_vision/configs/proj/uvim).
- [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013), by
  Lucas Beyer*, Pavel Izmailov*, Alexander Kolesnikov*, Mathilde Caron*, Simon
  Kornblith*, Xiaohua Zhai*, Matthias Minderer*, Michael Tschannen*, Ibrahim
  Alabdulmohsin*, Filip Pavetic*\
  Resources: [readme](big_vision/configs/proj/flexivit/README.md), [configs](big_vision/configs/proj/flexivit).
- [Dual PatchNorm](https://arxiv.org/abs/2302.01327), by Manoj Kumar, Mostafa Dehghani, Neil Houlsby.
- [Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design](https://arxiv.org/abs/2305.13035), by
  Ibrahim Alabdulmohsin*, Xiaohua Zhai*, Alexander Kolesnikov, Lucas Beyer*.
- (partial) [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442), by
  Mostafa Dehghani*, Josip Djolonga*, Basil Mustafa*, Piotr Padlewski*, Jonathan Heek*, *wow many middle authors*, Neil Houlsby*.
- (partial) [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505), by
  Fabian Mentzer, David Minnen, Eirikur Agustsson, Michael Tschannen.
- [GIVT: Generative Infinite-Vocabulary Transformers](https://arxiv.org/abs/2312.02116), by
  Michael Tschannen, Cian Eastwood, Fabian Mentzer\
  Resources: [readme](big_vision/configs/proj/givt/README.md), [config](big_vision/configs/proj/givt/givt_imagenet2012.py), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/givt/givt_demo_colab.ipynb).

### Multimodal research

- [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991), by
  Xiaohua Zhai*, Xiao Wang*, Basil Mustafa*, Andreas Steiner*, Daniel Keysers,
  Alexander Kolesnikov, and Lucas Beyer*\
  Resources: [trainer](big_vision/trainers/proj/image_text/contrastive.py), [config](big_vision/configs/proj/image_text/lit_coco.py), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/lit.ipynb).
- [Image-and-Language Understanding from Pixels Only](https://arxiv.org/abs/2212.08045), by
  Michael Tschannen, Basil Mustafa, Neil Houlsby\
  Resources: [readme](big_vision/configs/proj/clippo/README.md), [config](big_vision/configs/proj/clippo/train_clippo.py), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/clippo/clippo_colab.ipynb).
- [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343), by
  Xiaohua Zhai*, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer*\
  Resources: [colab and models](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb), code TODO.
- [A Study of Autoregressive Decoders for Multi-Tasking in Computer Vision](https://arxiv.org/abs/2303.17376), by
  Lucas Beyer*, Bo Wan*, Gagan Madan*, Filip Pavetic*, Andreas Steiner*, Alexander Kolesnikov, André Susano Pinto, Emanuele Bugliarello, Xiao Wang, Qihang Yu, Liang-Chieh Chen, Xiaohua Zhai*.
- [Image Captioners Are Scalable Vision Learners Too](https://arxiv.org/abs/2306.07915), by
  Michael Tschannen*, Manoj Kumar*, Andreas Steiner*, Xiaohua Zhai, Neil Houlsby, Lucas Beyer*.\
  Resources: [readme](big_vision/configs/proj/cappa/README.md), [config](big_vision/configs/proj/cappa/pretrain.py), [model](big_vision/models/proj/cappa/cappa.py).
- [Three Towers: Flexible Contrastive Learning with Pretrained Image Models](https://arxiv.org/abs/2305.16999), by Jannik Kossen, Mark Collier, Basil Mustafa, Xiao Wang, Xiaohua Zhai, Lucas Beyer, Andreas Steiner, Jesse Berent, Rodolphe Jenatton, Efi Kokiopoulou.
- (partial) [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794), by Xi Chen, Xiao Wang, Soravit Changpinyo, *wow so many middle authors*, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, Radu Soricut.
- (partial) [PaLI-3 Vision Language Models: Smaller, Faster, Stronger](https://arxiv.org/abs/2310.09199), by Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, Radu Soricut.

### Training

- [Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237), by
  Lucas Beyer*, Xiaohua Zhai*, Amélie Royer*, Larisa Markeeva*, Rohan Anil,
  and Alexander Kolesnikov*\
  Resources: [README](big_vision/configs/proj/distill/README.md), [trainer](big_vision/trainers/proj/distill/distill.py), [colab](https://colab.research.google.com/drive/1nMykzUzsfQ_uAxfj3k35DYsATnG_knPl?usp=sharing).
- [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412), by
  Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
- [Surrogate Gap Minimization Improves Sharpness-Aware Training](https://arxiv.org/abs/2203.08065), by Juntang Zhuang, Boqing Gong, Liangzhe Yuan, Yin Cui, Hartwig Adam, Nicha Dvornek, Sekhar Tatikonda, James Duncan and Ting Liu \
  Resources: [trainer](big_vision/trainers/proj/gsam/gsam.py), [config](big_vision/configs/proj/gsam/vit_i1k_gsam_no_aug.py) [reproduced results](https://github.com/google-research/big_vision/pull/8#pullrequestreview-1078557411)
- [Tuning computer vision models with task rewards](https://arxiv.org/abs/2302.08242), by
  André Susano Pinto*, Alexander Kolesnikov*, Yuge Shi, Lucas Beyer, Xiaohua Zhai.
- (partial) [VeLO: Training Versatile Learned Optimizers by Scaling Up](https://arxiv.org/abs/2211.09760) by
  Luke Metz, James Harrison, C. Daniel Freeman, Amil Merchant, Lucas Beyer, James Bradbury, Naman Agrawal, Ben Poole, Igor Mordatch, Adam Roberts, Jascha Sohl-Dickstein.

### Misc

- [Are we done with ImageNet?](https://arxiv.org/abs/2006.07159), by
  Lucas Beyer*, Olivier J. Hénaff*, Alexander Kolesnikov*, Xiaohua Zhai*,
  and Aäron van den Oord*

# Codebase high-level organization and principles in a nutshell

The main entry point is a trainer module, which typically does all the
boilerplate related to creating a model and an optimizer, loading the data,
checkpointing and training/evaluating the model inside a loop. We provide the
canonical trainer `train.py` in the root folder. Normally, individual projects
within `big_vision` fork and customize this trainer.

All models, evaluators and preprocessing operations live in the corresponding
subdirectories and can often be reused between different projects. We encourage
compatible APIs within these directories to facilitate reusability, but it is
not strictly enforced, as individual projects may need to introduce their custom
APIs.

We have a powerful configuration system, with the configs living in the
`configs/` directory. Custom trainers and modules can directly extend/modify
the configuration options.

Project-specific code resides in the `.../proj/...` namespace. It is not always
possible to keep project-specific in sync with the core `big_vision` libraries,
Below we provide the [last known commit](#project-specific-commits)
for each project where the project code is expected to work.

Training jobs are robust to interruptions and will resume seamlessly from the
last saved checkpoint (assuming a user provides the correct `--workdir` path).

Each configuration file contains a comment at the top with a `COMMAND` snippet
to run it, and some hint of expected runtime and results. See below for more
details, but generally speaking, running on a GPU machine involves calling
`python -m COMMAND` while running on TPUs, including multi-host, involves

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all
  --command "bash big_vision/run_tpu.sh COMMAND"
```

See instructions below for more details on how to run `big_vision` code on a
GPU machine or Google Cloud TPU.

By default we write checkpoints and logfiles. The logfiles are a list of JSON
objects, and we provide a short and straightforward [example colab to read
and display the logs and checkpoints](https://colab.research.google.com/drive/1R_lvV542WUp8Q2y8sbyooZOGCplkn7KI?usp=sharing).

# Current and future contents

The first release contains the core part of pre-training, transferring, and
evaluating classification models at scale on Cloud TPU VMs.

We have since added the following key features and projects:
- Contrastive Image-Text model training and evaluation as in LiT and CLIP.
- Patient and consistent distillation.
- Scaling ViT.
- MLP-Mixer.
- UViM.

Features and projects we plan to release in the near future, in no particular
order:
- ImageNet-21k in TFDS.
- Loading misc public models used in our publications (NFNet, MoCov3, DINO).
- Memory-efficient Polyak-averaging implementation.
- Advanced JAX compute and memory profiling. We are using internal tools for
    this, but may eventually add support for the publicly available ones.

We will continue releasing code of our future publications developed within
`big_vision` here.

### Non-content

The following exist in the internal variant of this codebase, and there is no
plan for their release:
- Regular regression tests for both quality and speed. They rely heavily on
    internal infrastructure.
- Advanced logging, monitoring, and plotting of experiments. This also relies
    heavily on internal infrastructure. However, we are open to ideas on this
    and may add some in the future, especially if implemented in a
    self-contained manner.
- Not yet published, ongoing research projects.


# GPU Setup

We first discuss how to setup and run `big_vision` on a (local) GPU machine,
and then discuss the setup for Cloud TPUs. Note that data preparation step for
(local) GPU setup can be largely reused for the Cloud TPU setup. While the
instructions skip this for brevity, we highly recommend using a
[virtual environment](https://docs.python.org/3/library/venv.html) when
installing python dependencies.

## Setting up python packages

The first step is to checkout `big_vision` and install relevant python
dependencies:

```
git clone https://github.com/google-research/big_vision
cd big_vision/
pip3 install --upgrade pip
pip3 install -r big_vision/requirements.txt
```

The latest version of `jax` library can be fetched as

```
pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

You may need a different `jax` package, depending on CUDA and cuDNN libraries
installed on your machine. Please consult
[official jax documentation](https://github.com/google/jax#pip-installation-gpu-cuda)
for more information.

## Preparing tfds data

For unified and reproducible access to standard datasets we opted to use the
`tensorflow_datasets` (`tfds`) library. It requires each dataset to be
downloaded, preprocessed and then to be stored on a hard drive (or, if you use
"Google Cloud", preferably stored in a "GCP bucket".).

Many datasets can be downloaded and preprocessed automatically when used
for the first time. Nevertheless, we intentionally disable this feature and
recommend doing dataset preparation step separately, ahead of the first run. It
will make debugging easier if problems arise and some datasets, like
`imagenet2012`, require manually downloaded data.

Most of the datasets, e.g. `cifar100`, `oxford_iiit_pet` or `imagenet_v2`
can be fully automatically downloaded and prepared by running

```
cd big_vision/
python3 -m big_vision.tools.download_tfds_datasets cifar100 oxford_iiit_pet imagenet_v2
```

A full list of datasets is available at [this link](https://www.tensorflow.org/datasets/catalog/overview#all_datasets).

Some datasets, like `imagenet2012` or `imagenet2012_real`, require the data to
be downloaded manually and placed into `$TFDS_DATA_DIR/downloads/manual/`,
which defaults to `~/tensorflow_datasets/downloads/manual/`. For example, for
`imagenet2012` and `imagenet2012_real` one needs to place the official
`ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` files in that directory
and then run
`python3 -m big_vision.tools.download_tfds_datasets imagenet2012 imagenet2012_real`
(which may take ~1 hour).

If you use `Google Cloud` and, TPUs in particular, you can then upload
the preprocessed data (stored in `$TFDS_DATA_DIR`) to
"Google Cloud Bucket" and use the bucket on any of your (TPU) virtual
machines to access the data.

## Running on a GPU machine

Finally, after installing all python dependencies and preparing `tfds` data,
the user can run the job using config of their choice, e.g. to train `ViT-S/16`
model on ImageNet data, one should run the following command:

```
python3 -m big_vision.train --config big_vision/configs/vit_s16_i1k.py --workdir workdirs/`date '+%m-%d_%H%M'`
```

or to train MLP-Mixer-B/16, run (note the `gpu8` config param that reduces the default batch size and epoch count):

```
python3 -m big_vision.train --config big_vision/configs/mlp_mixer_i1k.py:gpu8 --workdir workdirs/`date '+%m-%d_%H%M'`
```

# Cloud TPU VM setup

## Create TPU VMs

To create a single machine with 8 TPU cores, follow the following Cloud TPU JAX
document:
https://cloud.google.com/tpu/docs/run-calculation-jax

To support large-scale vision research, more cores with multiple hosts are
recommended. Below we provide instructions on how to do it.

First, create some useful variables, which we be reused:

```
export NAME=<a name of the TPU deployment, e.g. my-tpu-machine>
export ZONE=<GCP geographical zone, e.g. europe-west4-a>
export GS_BUCKET_NAME=<Name of the storage bucket, e.g. my_bucket>
```

The following command line will create TPU VMs with 32 cores,
4 hosts.

```
gcloud compute tpus tpu-vm create $NAME --zone $ZONE --accelerator-type v3-32 --version tpu-ubuntu2204-base
```

## Install `big_vision` on TPU VMs

Fetch the `big_vision` repository, copy it to all TPU VM hosts, and install
dependencies.

```
git clone https://github.com/google-research/big_vision
gcloud compute tpus tpu-vm scp --recurse big_vision/big_vision $NAME: --zone=$ZONE --worker=all
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "bash big_vision/run_tpu.sh"
```

## Download and prepare TFDS datasets

We recommend preparing `tfds` data locally as described above and then uploading
the data to `Google Cloud` bucket. However, if you prefer, the datasets which
do not require manual downloads can be prepared automatically using a TPU
machine as described below. Note that TPU machines have only 100 GB of disk
space, and multihost TPU slices do not allow for external disks to be attached
in a write mode, so the instructions below may not work for preparing large
datasets. As yet another alternative, we provide instructions
[on how to prepare `tfds` data on CPU-only GCP machine](#preparing-tfds-data-on-a-standalone-gcp-cpu-machine).

Specifically, the seven TFDS datasets used during evaluations will be generated
under `~/tensorflow_datasets` on TPU machine with this command:

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=0 --command "TFDS_DATA_DIR=~/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.tools.download_tfds_datasets cifar10 cifar100 oxford_iiit_pet oxford_flowers102 cars196 dtd uc_merced"
```

You can then copy the datasets to GS bucket, to make them accessible to all TPU workers.

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=0 --command "rm -r ~/tensorflow_datasets/downloads && gsutil cp -r ~/tensorflow_datasets gs://$GS_BUCKET_NAME"
```

If you want to integrate other public or custom datasets, i.e. imagenet2012,
please follow [the official guideline](https://www.tensorflow.org/datasets/catalog/overview).

## Pre-trained models

For the full list of pre-trained models check out the `load` function defined in
the same module as the model code. And for example config on how to use these
models, see `configs/transfer.py`.

## Run the transfer script on TPU VMs

The following command line fine-tunes a pre-trained `vit-i21k-augreg-b/32` model
on `cifar10` dataset.

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/transfer.py:model=vit-i21k-augreg-b/32,dataset=cifar10,crop=resmall_crop --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'` --config.lr=0.03"
```

## Run the train script on TPU VMs

To train your own big_vision models on a large dataset,
e.g. `imagenet2012` ([prepare the TFDS dataset](https://www.tensorflow.org/datasets/catalog/imagenet2012)),
run the following command line.

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/bit_i1k.py  --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'`"
```

## FSDP training.

`big_vision` supports flexible parameter and model sharding strategies.
Currently, we support a popular FSDP sharding via a simple config change, see [this config example](big_vision/configs/transfer.py).
For example, to run FSDP finetuning of a pretrained ViT-L model, run the following command (possible adjusting batch size depending on your hardware):

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/transfer.py:model=vit-i21k-augreg-l/16,dataset=oxford_iiit_pet,crop=resmall_crop,fsdp=True,batch_size=256 --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'` --config.lr=0.03"
```

## Image-text training with SigLIP.

A minimal example that uses public `coco` captions data:

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.trainers.proj.image_text.siglip --config big_vision/configs/proj/image_text/siglip_lit_coco.py --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%Y-%m-%d_%H%M'`"
```



## Sometimes useful gcloud commands

- Destroy the TPU machines: `gcloud compute tpus tpu-vm delete $NAME --zone $ZONE`
- Remove all big_vision-related folders on all hosts: `gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'rm -rf ~/big_vision ~/bv_venv'`

## Preparing `tfds` data on a standalone GCP CPU machine.

First create a new machine and a disk (feel free to adjust exact machine type and disk settings/capacity):

```
export NAME_CPU_HOST=<A name of a CPU-only machine>
export NAME_DISK=<A name of a disk>
gcloud compute instances create $NAME_CPU_HOST --machine-type c3-standard-22 --zone $ZONE --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud
gcloud compute disks create $NAME_DISK --size 1000GB --zone $ZONE --type pd-balanced
```

Now attach the disk to the newly create machine:

```
gcloud compute instances attach-disk $NAME_CPU_HOST --disk $NAME_DISK --zone $ZONE
```

Next, `ssh` to the machine `gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE` and
[follow instructions to format and mount the disk](https://cloud.google.com/compute/docs/disks/format-mount-disk-linux).
Let's assume it was mounted to `/mnt/disks/tfds`.

Almost there, now clone and set up `big_vision`:

```
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "git clone https://github.com/google-research/big_vision.git && cd big_vision && sh big_vision/run_tpu.sh"
```

Finally, prepare the dataset (e.g. `coco_captions`) using the utility script and
copy the result to you google cloud bucket:

```
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "cd big_vision && TFDS_DATA_DIR=/mnt/disks/tfds/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.tools.download_tfds_datasets coco_captions"
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "rm -rf /mnt/disks/tfds/tensorflow_datasets/downloads && gsutil cp -r /mnt/disks/tfds/tensorflow_datasets gs://$GS_BUCKET_NAME"
```


# ViT baseline

We provide a well-tuned ViT-S/16 baseline in the config file named
`vit_s16_i1k.py`. It achieves 76.5% accuracy on ImageNet validation split in
90 epochs of training, being a strong and simple starting point for research
on the ViT models.

Please see our [arXiv note](https://arxiv.org/abs/2205.01580) for more details
and if this baseline happens to by useful for your research, consider citing

```
@article{vit_baseline,
  url = {https://arxiv.org/abs/2205.01580},
  author = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
  title = {Better plain ViT baselines for ImageNet-1k},
  journal={arXiv preprint arXiv:2205.01580},
  year = {2022},
}
```

Replicating the result on consumer-grade machine with single GPU requires gradient accumulation,
cache / buffer adjustments, and `jax.distributed.initialize()` guard implemented in this branch.
Also, the code is rather brittle w.r.t. python package versions even though `big_vision/requirements.txt`
doesn't pin them to a known working version. For reference here is my current `pip freeze` output:

```
absl-py==1.3.0
aiohttp==3.8.3
aioitertools==0.11.0
aiosignal==1.3.1
albumentations==1.3.0
antlr4-python3-runtime==4.9.3
anyio==4.0.0
appdirs==1.4.4
apturl==0.5.2
aqtp==0.7.2
argon2-cffi==21.1.0
array-record==0.5.0
arrow==1.3.0
astunparse==1.6.3
async-lru==2.0.4
async-timeout==4.0.2
attrs==22.1.0
azure-core==1.26.1
azure-storage-blob==12.14.1
Babel==2.13.1
backcall==0.2.0
bcrypt==3.2.0
beautifulsoup4==4.11.1
beniget==0.4.1
bleach==4.1.0
blinker==1.4
blis==0.7.10
blobfile==2.0.1
Bottleneck==1.3.2
braceexpand==0.1.7
Brlapi==0.8.3
Brotli==1.0.9
cachetools==5.2.0
catalogue==2.0.9
certifi==2023.7.22
cffi==1.15.1
chardet==3.0.4
charset-normalizer==2.1.1
chex==0.1.86
click==8.1.3
clip-benchmark==1.4.0
cloudpathlib==0.13.0
cloudpickle==3.0.0
clu @ git+https://github.com/google/CommonLoopUtils@613dcbbb1ad587584fc0fc53a0e7d727169b90d6
cmake==3.25.0
colorama==0.4.4
command-not-found==0.3
confection==0.1.0
contextlib2==21.6.0
contourpy==1.0.7
cryptography==39.0.2
cupshelpers==1.0
cycler==0.11.0
cymem==2.0.7
Cython==0.29.28
dataclasses==0.6
datasets==2.8.0
dbus-python==1.2.18
debugpy==1.8.0
decorator==4.4.2
defer==1.0.6
defusedxml==0.7.1
dill==0.3.6
distlib==0.3.4
distrax==0.1.5
distro==1.7.0
distro-info==1.1+ubuntu0.2
dm-tree==0.1.7
docker-pycreds==0.4.0
docutils==0.16
duplicity==0.8.21
einops==0.7.0
entrypoints==0.4
et-xmlfile==1.0.1
etils==1.7.0
exceptiongroup==1.1.2
ExifRead-nocycle==3.0.1
fasteners==0.14.1
fastjsonschema==2.18.1
fasttext==0.9.2
fasttext-langdetect==1.0.3
filelock==3.13.4
fire==0.4.0
flake8==4.0.1
flatbuffers==23.5.26
flax==0.8.2
flaxformer @ git+https://github.com/google/flaxformer@399ea3a85e9807ada653fd0de1a9de627eb0acde
fonttools==4.39.2
fqdn==1.5.1
frozenlist==1.3.3
fs==2.4.12
fsspec==2024.3.1
ftfy==6.1.1
future==0.18.0
gast==0.4.0
gcld3==3.0.13
gdown==4.5.3
gitdb==4.0.9
GitPython==3.1.40
google-api-core==2.11.0
google-auth==2.27.0
google-auth-oauthlib==1.2.0
google-cloud-core==2.3.2
google-cloud-storage==2.7.0
google-crc32c==1.5.0
google-pasta==0.2.0
google-resumable-media==2.4.0
googleapis-common-protos==1.57.0
grpcio==1.50.0
h5py==3.11.0
html5lib==1.1
httplib2==0.20.2
huggingface-hub==0.11.0
hydra-core==1.3.2
idna==3.4
imageio==2.22.4
imagenetv2-pytorch @ git+https://github.com/modestyachts/ImageNetV2_pytorch@14d4456c39fe7f02a665544dd9fc37c1a5f8b635
img2dataset==1.40.0
immutabledict==4.2.0
importlib-metadata==4.6.4
importlib-resources==5.10.0
iniconfig==2.0.0
ipykernel==6.7.0
ipython==7.31.1
ipython_genutils==0.2.0
ipywidgets==6.0.0
isodate==0.6.1
isoduration==20.11.0
jax==0.4.26
jaxlib==0.4.26+cuda12.cudnn89
jdcal==1.0
jedi==0.18.0
jeepney==0.7.1
Jinja2==3.1.3
jmespath==1.0.1
joblib==1.2.0
json5==0.9.14
jsonpointer==2.4
jsonschema==4.19.2
jsonschema-specifications==2023.7.1
jupyter-console==6.4.0
jupyter-events==0.9.0
jupyter-lsp==2.2.0
jupyter_client==8.6.0
jupyter_core==5.5.0
jupyter_server==2.10.0
jupyter_server_terminals==0.4.4
jupyterlab==4.0.8
jupyterlab-pygments==0.1.2
jupyterlab_server==2.25.1
kaleido==0.2.1
keras==2.15.0
Keras-Preprocessing==1.1.2
keyring==23.5.0
kiwisolver==1.4.4
langcodes==3.3.0
language-selector==0.1
launchpadlib==1.10.16
lazr.restfulclient==0.14.4
lazr.uri==1.0.6
lazy_loader==0.3
libclang==14.0.6
lightning-utilities==0.10.1
lion-pytorch==0.0.4
lit==15.0.7
littleutils==0.2.2
llvmlite==0.38.0
lockfile==0.12.2
loguru==0.7.2
louis==3.20.0
lxml==4.9.2
lz4==3.1.3+dfsg
macaroonbakery==1.3.1
Mako==1.1.3
Markdown==3.4.1
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.7.1
matplotlib-inline==0.1.3
mccabe==0.6.1
mdurl==0.1.2
mistune==3.0.2
ml-collections==0.1.1
ml-dtypes==0.2.0
mock==4.0.3
monotonic==1.6
more-itertools==8.10.0
mpmath==1.3.0
msgpack==1.0.8
msrest==0.7.1
multidict==6.0.3
multiprocess==0.70.14
murmurhash==1.0.9
namex==0.0.8
nbclient==0.5.6
nbconvert==7.11.0
nbformat==5.9.2
nest-asyncio==1.5.4
netifaces==0.11.0
networkx==3.3
nltk==3.8.1
nose==1.3.7
notebook==7.0.6
notebook_shim==0.2.3
numba==0.55.1
numexpr==2.8.1
numpy==1.26.4
nvidia-cublas-cu11==11.11.3.6
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvcc-cu12==12.4.131
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu11==8.7.0.84
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu11==10.9.0.58
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu11==10.3.0.86
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu11==11.7.5.86
nvidia-cusparse-cu12==12.1.0.106
nvidia-ml-py3==7.352.0
nvidia-nccl-cu11==2.19.3
nvidia-nccl-cu12==2.19.3
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu11==11.8.86
nvidia-nvtx-cu12==12.1.105
oauthlib==3.2.2
odfpy==1.4.2
ogb==1.3.5
olefile==0.46
omegaconf==2.3.0
open-clip-torch @ file:///home/jason-chou/Downloads/open_clip
opencv-python==4.6.0.66
opencv-python-headless==4.6.0.66
openpyxl==3.0.9
opt-einsum==3.3.0
optax==0.2.2
optree==0.11.0
orbax-checkpoint==0.5.10
outdated==0.2.2
overrides==7.4.0
packaging==21.3
pandas==1.5.1
pandocfilters==1.5.0
panopticapi @ git+https://github.com/akolesnikoff/panopticapi.git@a698a12deb21e4cf0f99ef0581b2c30c466bf355
parameterized==0.8.1
paramiko==2.9.3
parso==0.8.1
pathtools==0.1.2
pathy==0.10.2
patsy==0.5.3
pexpect==4.8.0
pickleshare==0.7.5
pillow==10.3.0
platformdirs==2.5.1
plotly==5.13.1
pluggy==1.2.0
ply==3.11
praw==7.1.0
prawcore==1.5.0
preshed==3.0.8
prometheus-client==0.9.0
promise==2.3
prompt-toolkit==3.0.28
protobuf==4.25.3
psutil==5.9.4
ptyprocess==0.7.0
py==1.10.0
pyarrow==7.0.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pybind11==2.10.1
pycairo==1.20.1
pycld3==0.22
pycocoevalcap==1.2
pycocotools==2.0.7
pycodestyle==2.8.0
pycparser==2.21
pycryptodomex==3.16.0
pycuda==2021.1
pycups==2.0.1
pydantic==1.10.12
pydot==1.4.2
pyflakes==2.4.0
Pygments==2.17.2
PyGObject==3.42.1
pygpu==0.7.6
PyJWT==2.3.0
pymacaroons==0.13.0
PyNaCl==1.5.0
pyparsing==3.0.9
pyRFC3339==1.1
pyrsistent==0.18.1
pysimdjson==5.0.2
PySocks==1.7.1
pytest==7.2.0
pytest-split==0.8.0
python-apt==2.4.0+ubuntu3
python-dateutil==2.8.2
python-debian==0.1.43+ubuntu1.1
python-json-logger==2.0.7
pythran==0.10.0
pytools==2021.2.8
pytorch-triton==2.1.0+6e4932cda8
pytz==2022.6
PyWavelets==1.4.1
pyxdg==0.27
PyYAML==5.4.1
pyzmq==25.1.1
qudida==0.0.4
referencing==0.30.2
regex==2022.10.31
reportlab==3.6.8
requests==2.28.1
requests-oauthlib==1.3.1
responses==0.18.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rich==13.7.1
rpds-py==0.12.0
rsa==4.7.2
safetensors==0.4.0
scann==1.3.1
scikit-cuda==0.5.3
scikit-image==0.19.3
scikit-learn==1.1.3
scipy==1.13.0
screen-resolution-extra==0.0.0
SecretStorage==3.3.1
Send2Trash==1.8.2
sentencepiece==0.1.97
sentry-sdk==1.11.1
setproctitle==1.3.2
shortuuid==1.0.11
simplejson==3.17.6
six==1.16.0
smart-open==6.3.0
smmap==5.0.0
sniffio==1.3.0
soupsieve==2.3.2.post1
spacy==3.6.0
spacy-legacy==3.0.12
spacy-loggers==1.0.4
srsly==2.4.7
statsmodels==0.13.5
style==1.1.0
sympy==1.12
systemd-python==234
tables==3.7.0
tenacity==8.2.2
tensorboard==2.15.2
tensorboard-data-server==0.7.2
tensorboard-plugin-wit==1.8.1
tensorflow==2.15.0
tensorflow-addons==0.23.0
tensorflow-cpu==2.15.0
tensorflow-datasets==4.9.4
tensorflow-estimator==2.15.0
tensorflow-gan==2.1.0
tensorflow-hub==0.16.1
tensorflow-io-gcs-filesystem==0.32.0
tensorflow-metadata==1.14.0
tensorflow-probability==0.24.0
tensorflow-text==2.15.0
tensorstore==0.1.56
termcolor==2.1.0
terminado==0.13.1
testpath==0.5.0
tf-keras==2.15.0
tfds-nightly==4.9.4.dev202404230044
Theano==1.0.5
thinc==8.1.10
threadpoolctl==3.1.0
tifffile==2022.10.10
tiktoken==0.1.2
timm==0.6.11
tinycss2==1.2.1
tokenizers==0.13.2
toml==0.10.2
tomli==2.0.1
toolz==0.12.0
torch==2.2.2
torchaudio==2.2.2
torchmetrics==1.3.1
torchvision==0.17.2
tornado==6.3.3
tqdm==4.64.1
traitlets==5.13.0
transformers==4.24.0
triton==2.2.0
typeguard==2.13.3
typer==0.9.0
types-python-dateutil==2.8.19.14
typing_extensions==4.11.0
ubuntu-drivers-common==0.0.0
ubuntu-pro-client==8001
ufoLib2==0.13.1
ufw==0.36.1
ujson==5.9.0
unattended-upgrades==0.1
unicodedata2==14.0.0
update==0.0.1
update-checker==0.18.0
uri-template==1.3.0
urllib3==1.26.12
usb-creator==0.3.7
virtualenv==20.13.0+ds
vit-pytorch==1.6.3
wadllib==1.3.6
wandb==0.16.6
wasabi==1.1.2
wcwidth==0.2.5
webcolors==1.13
webdataset==0.2.31
webencodings==0.5.1
websocket-client==1.6.1
Werkzeug==2.2.2
wget==3.2
widgetsnbextension==2.0.0
wilds==2.0.0
wordsegment==1.3.1
wrapt==1.14.1
xdg==5
xkit==0.0.0
xlwt==1.3.0
xxhash==3.2.0
yarl==1.8.2
zipp==3.10.0
```

# Project specific commits

The last known commit where the specific project code is expected to work. The
core code and configs are expected to work at head.

| Project    | Commit                                                                                        |
|------------|-----------------------------------------------------------------------------------------------|
| UViM       | https://github.com/google-research/big_vision/commit/21bd6ebe253f070f584d8b777ad76f4abce51bef |
| image_text | https://github.com/google-research/big_vision/commit/8921d5141504390a8a4f7b2dacb3b3c042237290 |
| distill    | https://github.com/google-research/big_vision/commit/2f3f493af048dbfd97555ff6060f31a0e686f17f |
| GSAM       | WIP                                                                                           |
| CLIPPO     | https://github.com/google-research/big_vision/commit/fd2d3bd2efc9d89ea959f16cd2f58ae8a495cd44 |
| CapPa      | https://github.com/google-research/big_vision/commit/7ace659452dee4b68547575352c022a2eef587a5 |
| GIVT       | https://github.com/google-research/big_vision/commit/0cb70881dd33b3343b769347dc19793c4994b8cb |

# Citing the codebase

If you found this codebase useful for your research, please consider using
the following BibTEX to cite it:

```
@misc{big_vision,
  author = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
  title = {Big Vision},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/google-research/big_vision}}
}
```

# Disclaimer

This is not an official Google Product.

# License

Unless explicitly noted otherwise, everything in the big_vision codebase
(including models and colabs) is released under the Apache2 license.
See the LICENSE file for the full license text.
