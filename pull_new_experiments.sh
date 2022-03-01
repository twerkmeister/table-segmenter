#!/bin/bash

git pull
for remote in $(git branch -r | grep -v '\->' | grep "origin/experiment_"); do
  git branch --track "${remote#origin/}" "$remote"
done