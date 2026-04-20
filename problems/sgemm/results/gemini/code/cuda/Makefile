CXX = g++
NVCC = nvcc
LDFLAGS = -lm

SRCS = main.cc io.cc
CUDA_SRCS = sgemm_kernel.cu
OBJS = main.o io.o sgemm_kernel.o
TARGET = sgemm

INPUTDIR = medium/input
OUTPUTDIR = medium/output

MATA = $(INPUTDIR)/matrix1.txt
MATB = $(INPUTDIR)/matrix2t.txt
MATBT = $(INPUTDIR)/matrix2t.txt
OUTFILE = output.txt
EXPECTED = $(OUTPUTDIR)/matrix3.txt

all: $(TARGET)

main.o: main.cc
	$(CXX) -c main.cc -o main.o

io.o: io.cc
	$(CXX) -c io.cc -o io.o

sgemm_kernel.o: sgemm_kernel.cu
	$(NVCC) -c sgemm_kernel.cu -o sgemm_kernel.o

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(MATA) $(MATB) $(MATBT) $(OUTFILE)

clean: 
	rm -f $(TARGET) $(OUTFILE) $(OBJS)

.PHONY: all run clean
