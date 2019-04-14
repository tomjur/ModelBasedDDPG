#!/usr/bin/env bash
set -eo pipefail
# set -xv # debug

if [ "$(uname -s)" != "Darwin" ]; then echo "OSX not detected, aborting."; exit 1; fi

# needed for OSX
function realpath() { python -c "import os; print(os.path.realpath('$1'))"; }

project_root=$(realpath "${0%/*}/..")

# Build docker image
# Note, "$@" forwards script args to docker builder
export DOCKER_BUILDKIT=1
docker build "$@"\
    --secret id=id_rsa.pub,src="$HOME/.ssh/id_rsa.pub" \
    -t local-trajopt \
    -f "$project_root/Dockerfile.dev" \
    "$project_root"

# Create localtrajopt host, for secure secrets forwarding 
if [ -z "$(grep -E '1\s+localtrajopt' /etc/hosts)" ]; then
    echo "need host password to modify /etc/hosts"
    echo "127.0.0.1 localtrajopt" | sudo tee -a /etc/hosts;
fi

# Enable auto x forwardig and auto secrets forwarding
if [ -z "$(grep 'Host localtrajopt' ~/.ssh/config)" ]; then
    echo "# Odin docker config"
    echo "Host localtrajopt" >> ~/.ssh/config;
    echo "  # Auto X forwarding" >> ~/.ssh/config;
    echo "  XAuthLocation /opt/X11/bin/xauth" >> ~/.ssh/config;
    echo "  ForwardX11 yes" >> ~/.ssh/config;
    echo "  ForwardX11Trusted yes" >> ~/.ssh/config;
    echo "  # For convenience" >> ~/.ssh/config;
    echo "  StrictHostKeyChecking no" >> ~/.ssh/config;
fi

# Install XQuartz
if [ -z "$(brew cask list | grep xquartz)" ]; then
    brew cask install xquartz;
fi
if [ "$(defaults read org.macosforge.xquartz.X11 enable_iglx)" -ne "1" ]; then
    defaults write org.macosforge.xquartz.X11 enable_iglx -bool true;
fi