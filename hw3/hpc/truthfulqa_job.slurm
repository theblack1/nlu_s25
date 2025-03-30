#!/bin/bash

# Please adjust these settings according to your needs.
#sbatch --account=ds_ga_1012-2024sp
#sbatch --partition=n1s8-v100-1
#sbatch --nodes=1
#sbatch --ntasks-per-node=1
#sbatch --cpus-per-task=1
#sbatch --time=1:00:00
#sbatch --mem=20GB
#sbatch --gres=gpu:1
#sbatch --job-name=truthfulqa
#sbatch --mail-type=ALL
#sbatch --mail-user=fl2656@nyu.edu
#sbatch --output="truthfulqa.out"


module purge
# cd path/to/your/code
cd home/fl2656/nlu_s25/hw3
# OVERLAY_FILE=path/to/your/singularity/overlay/file
OVERLAY_FILE=path/to/your/singularity/overlay/file
SINGULARITY_IMAGE=path/to/your/singularity/image/file
# singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; bash hpc/run_truthfulqa_evaluation.sh"
singularity exec --nv --bind /scratch/fl2656/ --bind /share/apps/llama:/share/apps/llama --overlay /scratch/fl2656/overlay-25GB-500K.ext3:rw /scratch/fl2656/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "source /ext3/env.sh; bash /home/fl2656/nlu_s25/hw3/hpc/run_truthfulqa_evaluation.sh"