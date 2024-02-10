#!/bin/sh
#
# Script for training thalamocortical models.
#
#SBATCH --account=theory         # Replace ACCOUNT with your group account name
#SBATCH --job-name=reinforce     # The job name.
#SBATCH -c 4                     # The number of cpu cores to use
#SBATCH -N 1                     # The number of nodes to use
#SBATCH -t 0-3:00                # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb        # The memory the job will use per cpu core

module load anaconda

#Command to execute Python program
python main_batch.py 'motor_reinforce' $SLURM_ARRAY_TASK_ID
#python main_batch.py 'motor_reinforce_rflo' $SLURM_ARRAY_TASK_ID

#End of script
