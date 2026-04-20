# ---- Compilers ----
CXX      = g++
CXXFLAGS = -O2 -Wall
LDFLAGS  = -lm

NVCC      = nvcc
NVCCFLAGS = -O2

# ---- Sources ----
CPU_SRCS   = main.cc io.cc
CPU_TARGET = sgemm

CUDA_SRCS   = main.cu sgemm_kernel.cu io.cc
CUDA_TARGET = sgemm_cuda

# ---- Test data ----
INPUTDIR  = medium/input
OUTPUTDIR = medium/output
MATA      = $(INPUTDIR)/matrix1.txt
MATB      = $(INPUTDIR)/matrix2t.txt
MATBT     = $(INPUTDIR)/matrix2t.txt
OUTFILE   = output.txt
EXPECTED  = $(OUTPUTDIR)/matrix3.txt

# ---- Targets ----
all: $(CPU_TARGET)

cuda: $(CUDA_TARGET)

$(CPU_TARGET): $(CPU_SRCS) sgemm_kernel.cc
	$(CXX) $(CXXFLAGS) -o $(CPU_TARGET) main.cc io.cc $(LDFLAGS)

$(CUDA_TARGET): $(CUDA_SRCS)
	$(NVCC) $(NVCCFLAGS) -o $(CUDA_TARGET) main.cu sgemm_kernel.cu io.cc

run: $(CPU_TARGET)
	./$(CPU_TARGET) $(MATA) $(MATB) $(MATBT) $(OUTFILE)

run_cuda: $(CUDA_TARGET)
	./$(CUDA_TARGET) $(MATA) $(MATB) $(MATBT) $(OUTFILE)

clean:
	rm -f $(CPU_TARGET) $(CUDA_TARGET) $(OUTFILE)

.PHONY: all cuda run run_cuda clean
