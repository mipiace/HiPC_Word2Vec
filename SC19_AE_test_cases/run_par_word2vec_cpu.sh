#!/bin/bash

data=../data/text8
binary=../PAR_Word2Vec_cpu

ncores=28 # set this to #logical cores of your machine (with hyper-threading if available)
niters=10

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_cpu_vectors_text8.txt -size 128 -window 8 -negative 5 -sample 1e-4 -threads $ncores -binary 0 -iter $niters -min-count 5 -save-vocab vocab_dpWord2Vec_CPU_text8.txt -batch-size 24