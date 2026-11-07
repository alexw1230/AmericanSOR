#!/bin/bash
#SBATCH --job-name=amer_psor
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -p short-96core
#SBATCH --output=psor_%j.out

module purge
module load gcc/13.2.0
module load openmpi/gcc13.2/4.1.6

set -euo pipefail

export OMPI_MCA_coll_hcoll_enable=0

cd "$SLURM_SUBMIT_DIR"

make -f Make/Makefile clean && make -f Make/Makefile

RES=results/${SLURM_JOB_ID}
mkdir -p "$RES"
echo "host: $(hostname)"   | tee    "$RES/env.txt"
module list 2>&1           | tee -a "$RES/env.txt"
mpicc --version 2>&1       | tee -a "$RES/env.txt"
lscpu                      >        "$RES/lscpu.txt"

MPIRUN="mpirun --bind-to core --map-by core"

#Timing + bitwise check, 3 reps each, M = 100 time steps
for N in 128 256 512 1024; do
  for P in 1 2 4 16 64; do
    $MPIRUN -np $P ./psor -n $N -m 100 -r 3 -o "$RES/american.csv"
  done
done

#European vs closed form, error should drop ~4x from M = 100 to 400
for M in 100 400; do
  $MPIRUN -np 16 ./psor -n 512 -m $M -e -o "$RES/european.csv"
done

echo "done"
