#!/bin/bash

#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:20:00
#SBATCH --mem=6000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module load Python/3.6.3-foss-2017b

CURR_PROG=train_mlp_numpy

cp -r $HOME/Deep_Learning/assignment_1/code $TMPDIR
cd $TMPDIR/code

srun python3 $TMPDIR/code/$CURR_PROG.py > $CURR_PROG-results.out

cp $CURR_PROG-results.out $HOME/Deep_Learning/assignment_1/results
cp $HOME/Deep_Learning/assignment_1/slurm*.out $HOME/Deep_Learning/assignment_1/results/
rm $HOME/Deep_Learning/assignment_1/slurm*.out 
