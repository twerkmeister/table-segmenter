#!/bin/bash

for experiment_branch in $(git branch | tr -d "* " | grep "^experiment_"); do
  experiment_name=${experiment_branch##experiment_}
  if [ -d "experiments/$experiment_name" ]; then
    continue
  fi
  git checkout "$experiment_branch"
  for run_i in {01..05}; do
    experiment_folder="experiments/$experiment_name/$run_i"
    mkdir -p "$experiment_folder"
    python -m table_segmenter.train ~/datasets/table_segmentation_data/train ~/datasets/table_segmentation_data/val "$experiment_folder"
  done
done