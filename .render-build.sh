#!/usr/bin/env bash

# Install Git LFS
apt-get update && apt-get install -y git-lfs

# Initialize Git LFS and pull files
git lfs install
git lfs pull

# Continue with normal build
pip install -r requirements.txt
