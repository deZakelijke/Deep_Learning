#!/bin/bash

#SBATCH --job-name=python_experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:15:00
#SBATCH --mem=6000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module load Python/3.6.3-foss-2017b

CURR_PROG=train_convnet_pytorch
#CURR_PROG=train_mlp_pytorch
LEARNING_RATE=1e-4
MAX_STEPS=5000
HIDDEN_UNITS="400,400,300,300,300"
BATCH_SIZE=32
OPTIM="Adam"

cp -r $HOME/Deep_Learning/assignment_1/code $TMPDIR
cd $TMPDIR/code

#srun python3 $TMPDIR/code/$CURR_PROG.py --learning_rate $LEARNING_RATE --max_steps $MAX_STEPS --dnn_hidden_units $HIDDEN_UNITS --batch_norm --optim $OPTIM --batch_size $BATCH_SIZE> $CURR_PROG-results.out
srun python3 $TMPDIR/code/$CURR_PROG.py --learning_rate $LEARNING_RATE --max_steps $MAX_STEPS  --batch_size $BATCH_SIZE --cuda > $CURR_PROG-results_default_settings.out

cp $CURR_PROG-results.out $HOME/Deep_Learning/assignment_1/results
cp *.pdf $HOME/Deep_Learning/assignment_1/results
cp $HOME/Deep_Learning/assignment_1/slurm*.out $HOME/Deep_Learning/assignment_1/results/
rm $HOME/Deep_Learning/assignment_1/slurm*.out 
