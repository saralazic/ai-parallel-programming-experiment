# Feynman-Kac 1D — Sequential (C) and Parallel (CUDA) builds

CC       = gcc
NVCC     = nvcc

CFLAGS   = -O2 -Wall
LDFLAGS  = -lm

# sm_60+ required for native double-precision atomicAdd
NVCC_ARCH   = -arch=sm_60
NVCC_FLAGS  = -O2 $(NVCC_ARCH)
NVCC_LIBS   = -lcurand

SEQ_SRC  = feynman-kac-seq.c
CUDA_SRC = feynman-kac-cuda.cu

SEQ_BIN  = feynman-kac-seq
CUDA_BIN = feynman-kac-cuda

.PHONY: all seq cuda clean

all: seq cuda

seq: $(SEQ_BIN)

cuda: $(CUDA_BIN)

$(SEQ_BIN): $(SEQ_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(CUDA_BIN): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)

clean:
	rm -f $(SEQ_BIN) $(CUDA_BIN)
