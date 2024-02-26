# directory config
DATADIR=/storage/datasets/imagenet
CKPTDIR=/storage/weak2strong/vision/ckpt/imagenet/alexnet1
EMBEDDIR=/storage/weak2strong/vision/embedding/imagenet

# supervisor config
EPOCH=0
ITER=1000
SOFT=False
RESULTDIR=result

# 1-fold generalist
# python multi_weak_strong_oracle.py \
# 	${CKPTDIR}/0_1000/head-${EPOCH}-${ITER}.pth.tar \
# 	--data_path ${DATADIR} \
# 	--embed_path ${EMBEDDIR} \
# 	--result_path ${RESULTDIR} \
# 	--soft_teacher ${SOFT} \
# 	--weak_model_name alexnet1 \
# 	--strong_model_name vits14_dino \
# 	--n_epochs 20 --lr 1e-4

# 2-fold specialist
python multi_weak_strong_oracle.py \
	${CKPTDIR}/0_500/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/500_1000/head-${EPOCH}-${ITER}.pth.tar \
	--data_path ${DATADIR} \
	--embed_path ${EMBEDDIR} \
	--result_path ${RESULTDIR} \
	--soft_teacher ${SOFT} \
	--weak_model_name alexnet1 \
	--strong_model_name vits14_dino \
	--n_epochs 20 --lr 1e-4

# 4-fold specialist
python multi_weak_strong_oracle.py \
	${CKPTDIR}/0_250/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/250_500/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/500_750/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/750_1000/head-${EPOCH}-${ITER}.pth.tar \
	--data_path ${DATADIR} \
	--embed_path ${EMBEDDIR} \
	--result_path ${RESULTDIR} \
	--soft_teacher ${SOFT} \
	--weak_model_name alexnet1 \
	--strong_model_name vits14_dino \
	--n_epochs 20 --lr 1e-4

# 48-fold specialist
python multi_weak_strong_oracle.py \
	${CKPTDIR}/0_125/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/125_250/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/250_375/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/375_500/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/500_625/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/625_750/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/750_875/head-${EPOCH}-${ITER}.pth.tar \
	${CKPTDIR}/875_1000/head-${EPOCH}-${ITER}.pth.tar \
	--data_path ${DATADIR} \
	--embed_path ${EMBEDDIR} \
	--result_path ${RESULTDIR} \
	--soft_teacher ${SOFT} \
	--weak_model_name alexnet1 \
	--strong_model_name vits14_dino \
	--n_epochs 20 --lr 1e-4
