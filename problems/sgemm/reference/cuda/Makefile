NVCC = nvcc
NVCCFLAGS ?= -O2 -arch=sm_86
LDFLAGS = -lm

SRCS = main.cu io.cc sgemm_kernel.cu
TARGET = sgemm

INPUTDIR = medium/input
MATA = $(INPUTDIR)/matrix1.txt
MATB = $(INPUTDIR)/matrix2t.txt
MATBT = $(INPUTDIR)/matrix2t.txt
OUTFILE = output.txt
EXPECTED = medium/output/matrix3.txt

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) main.cu io.cc sgemm_kernel.cu $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(MATA) $(MATB) $(MATBT) $(OUTFILE)

clean:
	rm -f $(TARGET) $(OUTFILE)

.PHONY: all run clean
