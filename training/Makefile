CC = g++
NVCC = nvcc
CXXFLAGS := -I ./ -O3 -std=c++11 -Wall

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -std=c++11 -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

all: main markovSeq

OBJS:= main.o markovcuda.o
main : $(OBJS)
	$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS)
main.o : main.cpp
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c $< -o $@
main.o: markov.h
markovcuda.o : markovcuda.cu markov.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c markovcuda.cu -o $@

markovSeq:  markovSeq.o
	$(CXX) $(CXXFLAGS) markovSeq.o -o markovSeq
markovSeq.o : markovSeq.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@
markovSeq.o: markov.h
clean:
	rm -f markovSeq markovSeq.o