#!/bin/bash
#SBATCH --job-name=copy_files
#SBATCH --partition=cpu_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --begin=now
#SBATCH --time=1-00:00:00
#SBATCH --mem=16GB
#SBATCH --mail-type=END
#SBATCH --mail-user=wenke.liu@nyulangone.org
#SBATCH --output=copy_files_%j.out
#SBATCH --error=copy_files_%j.error


cd /gpfs/data/proteomics/projects/Runyu/CPTAC-UCEC/images/POLE/tiles
scp -r * fenyopc002:/media/data02/POLE
