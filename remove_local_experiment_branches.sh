#!/bin/bash

git co master

for experiment_branch in $(git branch | tr -d "* " | grep "^experiment_"); do
  git branch -D "$experiment_branch"
done

git remote prune origin