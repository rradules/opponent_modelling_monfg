#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --mem=1gb
#SBATCH --output=logs/output-%A.out
#SBATCH --error=logs/err-%A.err
#SBATCH --job-name=baseline
#SBATCH --mail-type=END
#SBATCH --mail-user=roxana@ai.vub.ac.be

JOBDIR="$HOME/lola_monfg"
cd $JOBDIR

srun source activate nm
srun python iag_DICE.py -trials 10 -lookahead 5 -mooc 'SER'
