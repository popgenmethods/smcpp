#!/bin/bash
# Passes
smc++ posterior -v model.final.json out.npz chr11_5subjs.smc.gz
# Fails
smc++ posterior -v model.final.json out.npz chr11_5subjs_broken.smc.gz
