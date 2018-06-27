#!/bin/bash -x
SMC=$(which smc++)
TMP=$(mktemp -d)
set -e
$SMC vcf2smc -v example/example.vcf.gz $TMP/example.1.smc.gz 1 msp1:msp_0
$SMC vcf2smc -v example/example.vcf.gz $TMP/example.11.smc.gz 1 msp1:msp_1
$SMC vcf2smc -d msp_0 msp_0 example/example.vcf.gz $TMP/example.2.smc.gz 1 msp2:msp_0,msp_3,msp_4
$SMC vcf2smc -d msp_1 msp_1 example/example.vcf.gz $TMP/example.12.smc.gz 1 msp1:msp_1,msp_2 msp2:msp_3,msp_4,msp_0
$SMC chunk 10 200000 $TMP/chunk.1. $TMP/example.1.smc.gz
$SMC chunk 10 200000 $TMP/chunk.2. $TMP/example.2.smc.gz
$SMC chunk 10 200000 $TMP/chunk.12. $TMP/example.12.smc.gz
$SMC estimate --em-iterations 1 -o $TMP/out/1 --unfold --knots 5 --em-iterations 1 1.25e-8 $TMP/example.1.smc.gz
$SMC estimate --em-iterations 1 -o $TMP/out/1 --unfold --knots 5 --timepoints 33,1000 --em-iterations 1 1.25e-8 $TMP/example.1.smc.gz
$SMC estimate --em-iterations 1 -p 0.01 -r 1e-8 -o $TMP/out/2 --knots 5 --em-iterations 1 1.25e-8 $TMP/example.2.smc.gz
$SMC cv --em-iterations 1 --folds 2 -o $TMP/out/cv --fold 0 1e-8 $TMP/example.1.smc.gz $TMP/example.11.smc.gz
$SMC cv --em-iterations 1 --folds 2 -o $TMP/out/cv --fold 1 1e-8 $TMP/example.1.smc.gz $TMP/example.11.smc.gz
$SMC cv --em-iterations 1 --folds 2 -o $TMP/out/cv 1e-8 $TMP/example.1.smc.gz $TMP/example.11.smc.gz
$SMC cv --em-iterations 1 --folds 2 -o $TMP/out/cv1 1e-8 $TMP/example.1.smc.gz $TMP/example.11.smc.gz
$SMC split -o $TMP/out/split --em-iterations 1 \
    $TMP/out/1/model.final.json \
    $TMP/out/2/model.final.json \
    $TMP/example.*.smc.gz
$SMC split -o $TMP/out/split --em-iterations 1 \
    $TMP/out/1/model.final.json \
    $TMP/out/2/model.final.json \
    $TMP/chunk*
$SMC split --polarization-error .02 -o $TMP/out/split --em-iterations 1 \
    $TMP/out/1/model.final.json \
    $TMP/out/2/model.final.json \
    $TMP/example.*.smc.gz
$SMC posterior $TMP/out/1/model.final.json \
    $TMP/matrix.npz $TMP/example.1.smc.gz $TMP/example.1.smc.gz
$SMC simulate $TMP/out/1/model.final.json 2 .01 $TMP/out/1/sim.vcf
$SMC simulate $TMP/out/split/model.final.json 2 .01 $TMP/out/split/sim.vcf
$SMC posterior --colorbar -v \
    --heatmap $TMP/plot.png \
    $TMP/out/1/model.final.json \
    $TMP/matrix.npz $TMP/example.1.smc.gz
$SMC posterior -v \
    $TMP/out/split/model.final.json \
    $TMP/matrix.npz \
    $TMP/example.12.smc.gz
$SMC plot -c -g 29 $TMP/1.png $TMP/out/1/model.final.json
$SMC plot $TMP/2.pdf $TMP/out/2/model.final.json
$SMC plot -c $TMP/12.png $TMP/out/split/model.final.json
$SMC plot -c $TMP/all.pdf $TMP/out/*/model.final.json

