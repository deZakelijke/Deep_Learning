#!/bin/bash

#SBATCH --job-name=python_experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --mem=6000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module load Python/3.6.3-foss-2017b

#CURR_PROG=a3_vae_template
#CURR_PROG=a3_nf_template
CURR_PROG=a3_gan_template
DIR=code
#MODEL_TYPE='VAE'
#MODEL_TYPE='NF'
MODEL_TYPE='GAN'
NUM_CLASSES=10
BATCH_SIZE=128
LEARNING_RATE=1e-4
TRAIN_STEPS=50
MAX_NORM=10.0
DEVICE='cuda:0'
INPUT_LENGTH_STEPS=10
LATENT_DIM=100
EPOCHS=200


cp -r $HOME/Deep_Learning/assignment_3/$DIR $TMPDIR
cd $TMPDIR/$DIR
mkdir images


OUT_NAME=$CURR_PROG-results_$MODEL_TYPE.out
#srun python3 $TMPDIR/$DIR/$CURR_PROG.py --zdim $LATENT_DIM --epochs $EPCOHS > $OUT_NAME
srun python3 $TMPDIR/$DIR/$CURR_PROG.py --latent_dim $LATENT_DIM --n_epochs $EPOCHS > $OUT_NAME

mv $OUT_NAME $HOME/Deep_Learning/assignment_3/results
#mv LSTM_txt_$TXTFILE-temp_$TEMP.pt
#mv $MODEL_TYPE-model.pt $HOME/Deep_Learning/assignment_3/results/
mv $MODEL_TYPE-model_$LATENT_DIM.pt $HOME/Deep_Learning/assignment_3/results/

#mv elbo.pdf $HOME/Deep_Learning/assignment_3/results/
mv images/* $HOME/Deep_Learning/assignment_3/images/
#mv nfs_bpd.pdf $HOME/Deep_Learning/assignment_3/results/
mv $HOME/Deep_Learning/assignment_3/slurm*.out $HOME/Deep_Learning/assignment_3/results/
