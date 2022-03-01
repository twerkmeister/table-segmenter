#!/bin/bash

for experiment_branch in $(git branch | tr -d "* " | grep "^experiment_"); do
  git co "$experiment_branch"
  git push -u origin "$experiment_branch"
done
git co master