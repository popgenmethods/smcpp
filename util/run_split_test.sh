#!/bin/bash -x
SMC=$(which smc++)
set -e
LENGTH=100000000
./make_split.py $LENGTH > /tmp/split.vcf
bcftools view -Oz -o /tmp/split.vcf.gz /tmp/split.vcf
bcftools index /tmp/split.vcf.gz
$SMC vcf2smc --length $LENGTH -v /tmp/split.vcf.gz /tmp/example.1.smc.gz 1 msp1:$(echo msp_{0..9} | sed 's/ /,/g')
$SMC vcf2smc --length $LENGTH -v /tmp/split.vcf.gz /tmp/example.2.smc.gz 1 msp2:$(echo msp_{10..19} | sed 's/ /,/g')
$SMC vcf2smc --length $LENGTH -d msp_0 msp_0 /tmp/split.vcf.gz /tmp/example.12.smc.gz 1 msp1:$(echo msp_{0..9} | sed 's/ /,/g') msp2:$(echo msp_{10..19} | sed 's/ /,/g')
$SMC vcf2smc --length $LENGTH -d msp_10 msp_10 /tmp/split.vcf.gz /tmp/example.21.smc.gz 1 msp2:$(echo msp_{10..19} | sed 's/ /,/g') msp1:$(echo msp_{0..9} | sed 's/ /,/g')
$SMC estimate -o /tmp/out/1 -v 1e-8 /tmp/example.1.smc.gz --spline piecewise
$SMC estimate -o /tmp/out/2 -v 1e-8 /tmp/example.2.smc.gz --spline piecewise
$SMC split -o /tmp/out/split --em-iterations 1 -v \
    /tmp/out/1/model.final.json \
    /tmp/out/2/model.final.json \
    /tmp/example.12.smc.gz /tmp/example.21.smc.gz
