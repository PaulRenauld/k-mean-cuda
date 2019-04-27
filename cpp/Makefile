CXX=g++
EXECUTABLE := kmean-seq

CU_FILES   :=

CU_DEPS    :=

CC_FILES   := cxxopts.hpp kmean.cpp Point.cpp kmean_computer.cpp seq_computer.cpp silhouette_finder.cpp

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
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35
LIBS += GL glut cudart

ifneq ($(wildcard /opt/cuda-8.0/.*),)
# Latedays
LDFLAGS=-L/opt/cuda-8.0/lib64/ -lcudart
else
# GHC
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/kmean_computer.o $(OBJDIR)/Point.o $(OBJDIR)/seq_computer.o $(OBJDIR)/silhouette_finder.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
	mkdir -p $(OBJDIR)/

clean:
	rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)

check:	default
	./checker.pl

$(EXECUTABLE): $(OBJS) kmean.cpp 
	$(CXX) $(CXXFLAGS) -o $@ kmean.cpp $(OBJS)
# $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/cxxopts.o: 
	$(CXX) cxxopts.hpp $(CXXFLAGS) -c -o $@ 

$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

objs: $(OBJDIR)/all_objs.o

$(OBJDIR)/all_objs.o: dirs $(OBJS)
	$(CXX) $(HEADERS) $(CXXFLAGS) -o $@ $(OBJS) 
