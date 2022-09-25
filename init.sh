#!/usr/bin/env bash

mkdir -p src
pushd src || exit 1
git clone https://github.com/isl-org/MiDaS.git
git clone https://github.com/shariqfarooq123/AdaBins.git
git clone https://github.com/MSFTserver/pytorch3d-lite.git
popd || exit 1
