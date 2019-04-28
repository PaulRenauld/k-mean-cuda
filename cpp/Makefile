EXECUTABLES:= kmean-seq kmean-par

CU_FILES   := par_computer.cu

CU_DEPS    :=

CC_FILES   := kmean.cpp Point.cpp kmean_computer.cpp seq_computer.cpp silhouette_finder.cpp 

LOGS	   := logs

HEADERS    := Point.h data_structure.h kmean_computer.h seq_computer.h silhouette_finder.h

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g -std=c++11
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

#TODO switch to 61 and update ref
NVCCFLAGS=-std=c++11 -O3 -m64 --gpu-architecture compute_61
LIBS += GL glut cudart

LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=

SEQ_OBJS=$(OBJDIR)/silhouette_finder_seq.o $(OBJDIR)/seq_computer.o $(OBJDIR)/Point_seq.o $(OBJDIR)/kmean_computer_seq.o 

PAR_OBJS=$(OBJDIR)/silhouette_finder_par.o $(OBJDIR)/par_computer.o $(OBJDIR)/Point_par.o $(OBJDIR)/kmean_computer_par.o 


.PHONY: dirs clean

default: $(EXECUTABLES)

dirs:
	mkdir -p $(OBJDIR)/

clean:
	rm -rf $(OBJDIR) *~ $(EXECUTABLES) $(LOGS)

kmean-seq: $(OBJS) $(SEQ_OBJS) 
	$(CXX) $(CXXFLAGS) -DCUDA=0 -o $@ kmean.cpp $(OBJS) $(SEQ_OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

kmean-par: $(OBJS) $(PAR_OBJS) kmean.cpp 
	$(NVCC) $(NVCCFLAGS) -DCUDA=1 -o $@ kmean.cpp $(OBJS) $(PAR_OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%_seq.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -DCUDA=0 -c -o $@

$(OBJDIR)/%_par.o: %.cpp
	$(NVCC) $< $(NVCCFLAGS) -DCUDA=1 -c -o $@

$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -DCUDA=1 -c -o $@
