#!/bin/bash
#SBATCH --job-name=psor_p64
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH -p hbm-short-96core
#SBATCH --output=psor_p64_%j.out

module purge
module load gcc/13.2.0
module load openmpi/gcc13.2/4.1.6

set -euo pipefail

export OMPI_MCA_coll_hcoll_enable=0

cd "$SLURM_SUBMIT_DIR"

make -f Make/Makefile clean && make -f Make/Makefile

RES=results/${SLURM_JOB_ID}
mkdir -p "$RES"
echo "host: $(hostname)" | tee "$RES/env.txt"

#P = 64 under different rank placements, does the compute imbalance go away?
for MAP in core ppr:32:socket numa; do
  echo "=== map-by $MAP"
  mpirun --bind-to core --map-by $MAP --report-bindings -np 64 \
    ./psor -n 1024 -m 100 -r 3 -o "$RES/p64_${MAP%%:*}.csv" 2>&1 | tee -a "$RES/bindings.txt"
done

#48 ranks fit on one socket
mpirun --bind-to core --map-by core -np 48 ./psor -n 1024 -m 100 -r 3 -o "$RES/p48.csv"

echo "done"
