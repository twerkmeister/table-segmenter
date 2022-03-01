#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: $0 <prefix to put in front of the experiment branches>"
    exit 1
fi

for experiment_branch in $(git branch | tr -d "* " | grep "^experiment_"); do
  DONE_BRANCH_NAME="done_$1_$experiment_branch"
  git co "$experiment_branch"
  git co -b "$DONE_BRANCH_NAME"
  git push -u origin "$DONE_BRANCH_NAME"
  git push origin :"$experiment_branch"
done