#! /bin/bash

#$ -N multiagent_trips_regressions          # Give job a name
#$ -o scratch/freeapps.log                  # Log file
#$ -j y                                     # Merge error and output
#$ -l h_rt=1:00:00
#$ -pe omp 2

stata-mp -b do fit_rmb_models.do
