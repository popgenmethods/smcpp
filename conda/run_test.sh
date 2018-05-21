#!/bin/bash -x
SMC=$(which smc++)
set -e
$SMC vcf2smc -v example/example.vcf.gz /tmp/example.1.smc.gz 1 msp1:msp_0
$SMC vcf2smc -d msp_0 msp_0 example/example.vcf.gz /tmp/example.2.smc.gz 1 msp2:msp_0,msp_3,msp_4
$SMC vcf2smc -d msp_1 msp_1 example/example.vcf.gz /tmp/example.12.smc.gz 1 msp1:msp_1,msp_2 msp2:msp_3,msp_4,msp_0
$SMC chunk 10 10000 /tmp/chunk.1. /tmp/example.1.smc.gz
$SMC chunk 10 10000 /tmp/chunk.2. /tmp/example.2.smc.gz
$SMC chunk 10 10000 /tmp/chunk.12. /tmp/example.12.smc.gz
$SMC estimate -o /tmp/out/1 --unfold --knots 5 --em-iterations 1 1.25e-8 /tmp/example.1.smc.gz
$SMC estimate -o /tmp/out/1 --unfold --knots 5 --timepoints 33,1000 --em-iterations 1 1.25e-8 /tmp/example.1.smc.gz
$SMC estimate -p 0.01 -r 1e-8 -o /tmp/out/2 --knots 5 --em-iterations 1 1.25e-8 /tmp/example.2.smc.gz
$SMC split -o /tmp/out/split --em-iterations 1 \
    /tmp/out/1/model.final.json \
    /tmp/out/2/model.final.json \
    /tmp/example.*.smc.gz
$SMC split -o /tmp/out/split --em-iterations 1 \
    /tmp/out/1/model.final.json \
    /tmp/out/2/model.final.json \
    /tmp/chunk*
$SMC split --polarization-error .02 -o /tmp/out/split --em-iterations 1 \
    /tmp/out/1/model.final.json \
    /tmp/out/2/model.final.json \
    /tmp/example.*.smc.gz
$SMC posterior /tmp/out/1/model.final.json \
    /tmp/matrix.npz /tmp/example.1.smc.gz /tmp/example.1.smc.gz
$SMC simulate /tmp/out/1/model.final.json 2 1 /tmp/out/1/sim.vcf
$SMC simulate /tmp/out/split/model.final.json 2 1 /tmp/out/split/sim.vcf
$SMC posterior --colorbar -v \
    --heatmap /tmp/plot.png \
    /tmp/out/1/model.final.json \
    /tmp/matrix.npz /tmp/example.1.smc.gz
$SMC posterior -v \
    /tmp/out/split/model.final.json \
    /tmp/matrix.npz \
    /tmp/example.12.smc.gz
$SMC plot -c -g 29 /tmp/1.png /tmp/out/1/model.final.json
$SMC plot /tmp/2.pdf /tmp/out/2/model.final.json
$SMC plot -c /tmp/12.png /tmp/out/split/model.final.json
$SMC plot -c /tmp/all.pdf /tmp/out/*/model.final.json
