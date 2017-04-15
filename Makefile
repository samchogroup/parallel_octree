NVCC = nvcc
INCDIRS=-I/usr/local/cuda-6.5/include
EFILE = octree.out
OBJS = octree.o
CUDAFLAGS = -std=c++11 -arch=sm_35 -rdc=true

octree.out: $(OBJS)
	@echo "linking ..."
	$(NVCC) $(CUDAFLAGS) -o $(EFILE) $(OBJS)

octree.o: octree.h
	$(NVCC) $(CUDAFLAGS) -c octree.cu -o octree.o

clean:
	rm -f $(OBJS) $(EFILE)
