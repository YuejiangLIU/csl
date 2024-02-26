# directory config
DATADIR=/storage/datasets/imagenet
CKPTDIR=/storage/weak2strong/vision/ckpt/imagenet/alexnet1

# 1-fold
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 0 --end-imbal 1000 --savedir $CKPTDIR $DATADIR

# 2-fold
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 0 --end-imbal 500 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 500 --end-imbal 1000 --savedir $CKPTDIR $DATADIR

# 4-fold
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 0 --end-imbal 250 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 250 --end-imbal 500 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 500 --end-imbal 750 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 750 --end-imbal 1000 --savedir $CKPTDIR $DATADIR

# 8-fold
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 0 --end-imbal 125 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 125 --end-imbal 250 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 250 --end-imbal 375 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 375 --end-imbal 500 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 500 --end-imbal 625 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 625 --end-imbal 750 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 750 --end-imbal 875 --savedir $CKPTDIR $DATADIR
python prepare_imagenet.py -a alexnet --lr 1e-4 --batch-size 256 --workers 12 --epochs 20 --start-imbal 875 --end-imbal 1000 --savedir $CKPTDIR $DATADIR
