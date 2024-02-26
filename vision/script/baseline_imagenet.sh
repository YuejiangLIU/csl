# directory config
DATADIR=/storage/datasets/imagenet
CKPTDIR=/storage/weak2strong/vision/ckpt/imagenet/alexnet1
EMBEDDIR=/storage/weak2strong/vision/embedding/imagenet

# supervisor config
EPOCH=0
ITER=1000
SOFT=False

python single_weak_strong.py \
	--weak_path ${CKPTDIR}/0_1000/head-${EPOCH}-${ITER}.pth.tar \
	--data_path ${DATADIR} \
	--embed_path ${EMBEDDIR} \
	--result_path result/imagenet \
	--soft_teacher ${SOFT} \
	--weak_model_name alexnet1 \
	--strong_model_name vits14_dino \
	--lr 1e-4 --n_epochs 20
