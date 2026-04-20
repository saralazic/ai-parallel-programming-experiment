NVCC = nvcc
NVCCFLAGS = -O3 -std=c++14
LDFLAGS = -lm

SRCS = main.cc io.cc sgemm_kernel.cu
TARGET = sgemm

INPUTDIR = medium/input
OUTPUTDIR = medium/output
MATA = $(INPUTDIR)/matrix1.txt
MATB = $(INPUTDIR)/matrix2t.txt
MATBT = $(INPUTDIR)/matrix2t.txt
OUTFILE = output.txt
EXPECTED = $(OUTPUTDIR)/matrix3.txt

all: $(TARGET)

$(TARGET): $(SRCS) sgemm_kernel.h
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) $(MATA) $(MATB) $(MATBT) $(OUTFILE)

clean:
	rm -f $(TARGET) $(OUTFILE)

.PHONY: all run clean
