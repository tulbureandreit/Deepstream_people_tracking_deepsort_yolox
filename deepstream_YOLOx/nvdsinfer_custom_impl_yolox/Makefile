

#CUDA_VER?=
CUDA_VER=10.2
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I../../../../sources/includes -I/usr/local/cuda-$(CUDA_VER)/include -I/usr/include/opencv4 -I/usr/lib/gcc/aarch64-linux-gnu/7/include/
CXXFLAGS:= -fopenmp

LIBS:= -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs -L/usr/lib -lopencv_core
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group -fopenmp

INCS:= $(wildcard *.h)
SRCFILES:= nvdsparsebbox_yolox.cpp 

TARGET_LIB:= libnvdsinfer_custom_impl_yolox.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $(CXXFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(TARGET_LIB) : $(TARGET_OBJS)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
	rm -rf $(TARGET_OBJS)
