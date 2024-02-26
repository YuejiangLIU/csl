### Environment

```bash
pip install -r requirement.txt
```

### Download Dataset

Download ImageNet data, see torchvision for instructions; should contain files `ILSVRC2012_devkit_t12.tar.gz` and `ILSVRC2012_img_val.tar`

### Prepare Supervisor

```bash
bash script/prepare_imagenet.sh
```

### Download Pre-trained

Instead of prepare the weak supervisors through the script above, one can also download our pre-trained ones and the corresponding embeddings from [google drive](https://drive.google.com/drive/folders/1EA_TCZavnuJK3_NPvmE-23gUK8xzey8Z?usp=drive_link).

### Configure directories

Set directories in the scripts under the [script folder](script)

```bash
DATADIR=<basedir>/datasets/imagenet
CKPTDIR=<basedir>/ckpt/imagenet/alexnet
EMBEDDIR=<basedir>/embedding/imagenet
```
