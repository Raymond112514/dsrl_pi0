#!/bin/bash
# Submit the qf=20 baseline, PCA, and VAE sweeps.
# Each array task is one node and one seed.
#
# Node budget, filled in this order:
#   rail_gpu4_high    16
#   rail_gpu4_normal   8
#   rail_gpu4_low      8
#   rail_gpu4_lowest   the rest
#
# This grid is 20 nodes. Baseline (12) and PCA (4) fill high.
# The 4 VAE seeds land on normal.

set -euo pipefail

cd /global/home/users/r112358/dsrl_pi0
mkdir -p /global/home/users/r112358/logs logs

tasks=()
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
  tasks+=("examples/scripts/slurm_aloha_baseline_qf20.sh:${i}")
done
for i in 0 1 2 3; do
  tasks+=("examples/scripts/slurm_aloha_pca_qf20.sh:${i}")
done
for i in 0 1 2 3; do
  tasks+=("examples/scripts/slurm_aloha_vae_qf20.sh:${i}")
done

qos_names=(rail_gpu4_high rail_gpu4_normal rail_gpu4_low rail_gpu4_lowest)
qos_caps=(16 8 8 1000000)

declare -A group_ids
keys=()
qos_i=0
used=0
for task in "${tasks[@]}"; do
  while (( qos_i < ${#qos_caps[@]} - 1 && used >= qos_caps[qos_i] )); do
    qos_i=$((qos_i + 1))
    used=0
  done
  qos="${qos_names[$qos_i]}"
  script="${task%%:*}"
  id="${task##*:}"
  key="${qos}|${script}"
  if [[ -z "${group_ids[$key]+x}" ]]; then
    keys+=("${key}")
    group_ids[$key]="${id}"
  else
    group_ids[$key]="${group_ids[$key]},${id}"
  fi
  used=$((used + 1))
done

for key in "${keys[@]}"; do
  qos="${key%%|*}"
  script="${key#*|}"
  echo "sbatch --qos=${qos} --array=${group_ids[$key]} ${script}"
  sbatch --qos="${qos}" --array="${group_ids[$key]}" "${script}"
done

squeue -u r112358 -o '%.18i %.20j %.18q %.8T %.10M %R' | head -40
