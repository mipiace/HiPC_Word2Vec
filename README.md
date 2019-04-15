# PAR-Word2Vec

The C++ and CUDA implementations of **P**arallel **A**trraction-**R**epulsion based **Word2Vec** described in the paper titled, "Parallel Data-Local Training for Optimizing Word2Vec Embeddings for Word and Graph Embeddings".

The data pre-processing and data loading parts of code are based on the pWord2Vec implementation from [Intel](https://github.com/IntelLabs/pWord2Vec).

## Dependencies
- Intel Compiler (The C++ code is optimized on Intel CPUs)
- CUDA Compiler (The CUDA code is optimized on NVIDIA Tesla P100 PCIE GPU)
- OpenMP (No separated installation is needed once Intel compiler is installed)
- MKL (The latest version "16.0.0 or higher" is preferred as it has been improved significantly in recent years)
- Python (tested on 3.6)
- NumPy
- Keras
- _Keras backend; default Tensorflow_
- A few other miscellaneous libraries packaged up in the `dependencies` directory for model quality evaluation
  
## Prepare seven datasets
1. Download the text8 and 1B-Word datasets: cd `data`; .\getText8.sh and .\getBillion.sh
2. Download the BlogCatalog graph dataset: cd `data`; wget https://www.dropbox.com/s/5sdqv854ioody6i/blog_catalog_random_walks
3. Download the ASTRO-Ph graph dataset: cd `data`; wget https://www.dropbox.com/s/efh8e6wyhu5n3so/ASTRO_PH_random_walks
4. Other three graph datasets are already included in `data` directory.

## Compile the codes
1. Compile the codes for PAR-Word2Vec-cpu, PAR-Word2Vec-gpu, word2vec-cpu, pWord2Vec-cpu, pSGNScc-cpu and accSGNS-cpu: execute make
2. Compile the codes only for wombat benchmark: cd `wombat`; execute make

## Validate the results in the Experimental Evaluation section of the paper
1. The directory `SC19_AE_test_cases` contains BASH test scripts for validating all the results in our SC19 submission. A pretrained word embedding text file is used for each individual evaluation task. Each test script validates one Figure or Table presented in the Experimental Evaluation section of the paper. The name of each test script corresponds to the Figure or Table number in the paper it validates.
  + To validate the results in Table8, execute `Table8.sh`.
  + To validate the results in Table5, execute `Table5.sh`.
