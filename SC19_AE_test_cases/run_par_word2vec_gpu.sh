#!/bin/bash

# Text datasets
data=../data/text8
# data=../data/1b

# Labeled graph datasets
# data=../data/blog_walk
# data=../data/ppi_walk
# data=../data/wiki_label_walk

# Unlabeled graph datasets
# data=../data/facebook_walk
# data=../data/CA-AstroPh_walk

binary=../PAR_Word2Vec_gpu

niters=10

export KMP_AFFINITY=explicit,proclist=[0-$(($ncores-1))],granularity=fine
numactl --interleave=all $binary -train $data -output PAR_Word2Vec_gpu_vectors_text8.txt -size 128 -window 8 -negative 5 -sample 1e-4 -binary 0 -alpha 0.025 -iter $niters -min-count 5 -save-vocab vocab_PAR_Word2Vec_gpu.txt -batch-size 24 -gamma 64 -max-num-sen 9400

# // Text datasets
# MAX_NUM_SENTENCES 9400    // PAR_Word2Vec_gpu_vectors_text8
# MAX_NUM_SENTENCES 30700000  // PAR_Word2Vec_gpu_vectors_1b

# // Labeled graph datasets
# MAX_NUM_SENTENCES 103500	// PAR_Word2Vec_gpu_vectors_blog_walk
# MAX_NUM_SENTENCES 39500	// PAR_Word2Vec_gpu_vectors_ppi_walk
# MAX_NUM_SENTENCES 48000	// PAR_Word2Vec_gpu_vectors_wiki_label_walk

# // Unlabeled graph datasets
# MAX_NUM_SENTENCES 41000 	// PAR_Word2Vec_gpu_vectors_facebook_walk
# MAX_NUM_SENTENCES 188000 	// PAR_Word2Vec_gpu_vectors_CA-AstroPh_walk