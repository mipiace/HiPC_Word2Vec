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
  
## Data preparation
1. Download the data: cd `data`; .\getText8.sh and .\getBillion.sh
2. Download the ASTRO-Ph data: cd `data`; wget https://www.dropbox.com/s/efh8e6wyhu5n3so/ASTRO_PH_random_walks
3. Download the BlogCatalog data: cd `data`; wget https://www.dropbox.com/s/5sdqv854ioody6i/blog_catalog_random_walks

## Compile the codes
1. Compile the codes for PAR-Word2Vec-cpu, PAR-Word2Vec-gpu, word2vec-cpu, pWord2Vec-cpu, pSGNScc-cpu and accSGNS-cpu: execute make
2. Compile the codes only for wombat benchmark: cd `wombat`; execute make

## Validate the results in the Experimental Evaluation section of the paper
1. The directory `SC19_AE_test_cases` contains BASH test scripts for validating all the results in our SC19 submission. Each test script validates one Figure or Table presented in the Experimental Evaluation section of the paper. The name of each test script corresponds to the Figure or Table number in the paper it validates.
  + To validate the results in Table8, execute `Table8.sh`.
  + To validate the results in Table5, execute `Table5.sh`.
  + A pretrained word embedding text file is used for each individual evaluation task.

## Intrinsic-Evaluation-tasks

Similarity and relatedness experiments for word embeddings, implemented in Python.

Pre-configured with 5 common datasets (citations provided below):

- [WordSim-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)
- [SimLex-999](https://fh295.github.io/simlex.html)

Default scoring model provided uses cosine similarity of embedding vectors.

A demo script `demo.sh` is provided, which will download a small set of word embeddings and run them through the experiments.

### References

```
Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin, "Placing Search in Context: The Concept Revisited", ACM Transactions on Information Systems, 20(1):116-131, January 2002

Eneko Agirre, Enrique Alfonseca, Keith Hall, Jana Kravalova, Marius Pasca, Aitor Soroa, A Study on Similarity and Relatedness Using Distributional and WordNet-based Approaches, In Proceedings of NAACL-HLT 2009.

SimLex-999: Evaluating Semantic Models with (Genuine) Similarity Estimation. 2014. Felix Hill, Roi Reichart and Anna Korhonen.
```


## Extrinsic-Evaluation-tasks

Fork of [shashwath94/Extrinsic-Evaluation-tasks](https://github.com/shashwath94/Extrinsic-Evaluation-tasks), with some cleaning to support running all tasks on an arbitrary set of embeddings more easily.

Original repository assembled for Rogers et al. (2018) "[What's in Your Embedding, And How It Predicts Task Performance](http://aclweb.org/anthology/C18-1228)".

This version of repository assembled and released due to usage in Whitaker et al. (2019) "[Characterizing the impact of geometric properties of word embeddings on task performance](https://arxiv.org/abs/1904.04866)".

### References

Bib entry for Rogers et al (2018):
```
@inproceedings{C18-1228,
  title = "What{'}s in Your Embedding, And How It Predicts Task Performance",
  author = "Rogers, Anna  and Hosur Ananthakrishna, Shashwath  and Rumshisky, Anna",
  booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
  month = aug,
  year = "2018",
  address = "Santa Fe, New Mexico, USA",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/C18-1228",
  pages = "2690--2703",
}
```

Bib entry for Whitaker et al (2019):
```
@inproceedings{Whitaker2019,
  title = "Characterizing the impact of geometric properties of word embeddings on task performance",
  author = "Whitaker, Brendan and Newman-Griffis, Denis and Haldar, Aparajita and Ferhatosmanoglu, Hakan and Fosler-Lussier, Eric",
  booktitle = "Proceedings of the 3rd Workshop on Evaluating Vector Space Representations for NLP",
  month = jun,
  year = "2019",
  address = "Minneapolis, Minnesota, USA",
  publisher = "Association for Computational Linguistics"
}
```

### README

To run all tasks, execute `run_tasks.sh`.

For each individual task, `preprocess.py` (where present) will load the preprocessed version of the dataset.
To train the model, run `train.py`

A pretrained word embedding text file is needed where every line has a word string followed by a space and the embedding vector.
For example, `acrobat 0.6056159735 -0.1367940009 -0.0936380029 0.8406270146 0.2641879916 0.4209069908 0.0607739985 0.5985950232 -1.1451450586 -0.8666719794 -0.5021889806 0.4398249984 0.9671009779 0.7413169742 -0.0954160020 -1.1526989937 -0.3915260136 -0.1520590037 0.0893440023 -0.2578850091 -0.6204599738 -0.8789629936 0.3581469953 0.5509790182 0.1234730035`

Data for NLI task can be found [here](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)

For sequence labeling tasks (POS, NER and chunking), please refer to [this repo](https://github.com/shashwath94/Sequence-Labeling)

