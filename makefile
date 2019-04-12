CXX = icc
CC = icc
CPPFLAGS = -std=c++11 -qopenmp -O3 -D USE_MKL -mkl=sequential -Wall -xhost
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

GPU_ARCH_FLAG = arch=compute_60,code=sm_60
NVCC_HOME = /usr/local/cuda/9.2.88
NVCC = nvcc
CUDA_INC = -I$(NVCC_HOME)/include
CUDA_LIB = -L$(NVCC_HOME)/lib64 -lcudart -lcublas -lcusparse
CUDA_FLAGS = -O3 -std=c++11 -m64 -gencode $(GPU_ARCH_FLAG)

all: PAR_Word2Vec_cpu PAR_Word2Vec_gpu

PAR_Word2Vec_cpu: PAR_Word2Vec_cpu.cpp
	$(CXX) PAR_Word2Vec_cpu.cpp -o PAR_Word2Vec_cpu $(CPPFLAGS)
PAR_Word2Vec_gpu: PAR_Word2Vec_gpu.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o PAR_Word2Vec_gpu PAR_Word2Vec_gpu.cu $(CUDA_LIB)

# word2vec_SG : word2vec_SG.c
# 	$(CC) word2vec_SG.c -o word2vec_SG $(CFLAGS)
# pWord2Vec: pWord2Vec.cpp
# 	$(CXX) pWord2Vec.cpp -o pWord2Vec $(CPPFLAGS)
# pSGNScc: pSGNScc.cpp
# 	$(CXX) pSGNScc.cpp -o pSGNScc $(CPPFLAGS)
# dpWord2Vec_CPU: dpWord2Vec_CPU.cpp
# 	$(CXX) dpWord2Vec_CPU.cpp -o dpWord2Vec_CPU $(CPPFLAGS)
# word2vec_gpu: word2vec_gpu.cu
# 	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o word2vec_gpu word2vec_gpu.cu $(CUDA_LIB)
# PAR_Word2Vec_GPU: PAR_Word2Vec_GPU.cu
# 	$(NVCC) $(CUDA_FLAGS) $(CUDA_INC) -o PAR_Word2Vec_GPU PAR_Word2Vec_GPU.cu $(CUDA_LIB)
clean:
	rm -rf PAR_Word2Vec_cpu PAR_Word2Vec_gpu

