CXX = g++
LDFLAGS = -lm

SRCS = main.cc io.cc
TARGET = sgemm

INPUTDIR = medium/input
OUTPUTDIR = medium/output
CXX = g++
LDFLAGS = -lm
SRCS = main.cc io.cc
TARGET = sgemm
INPUTDIR = medium/input
MATA = $(INPUTDIR)/matrix1.txt
MATB = $(INPUTDIR)/matrix2t.txt
MATBT = $(INPUTDIR)/matrix2t.txt
OUTFILE = output.txt
EXPECTED = $(OUTPUTDIR)/matrix3.txt

all: $(TARGET)

$(TARGET): $(SRCS) sgemm_kernel.cc
    $(CXX) $(CXXFLAGS) -o $(TARGET) main.cc io.cc $(LDFLAGS)

run: $(TARGET)
    ./$(TARGET) $(MATA) $(MATB) $(MATBT) $(OUTFILE)


clean: 
    rm -f $(TARGET) $(OUTFILE)

.PHONY: all run clean
