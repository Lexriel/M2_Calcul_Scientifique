
TARGET= TP


all: exe

FLAGS =

### Select ARCH

HOST= $(shell hostname)
ARCH = nvidia

ifeq ($(HOST), gpu285)
  ARCH = nvidia
endif

ifeq ($(HOST), gpu4890)
  ARCH = ati
endif

ifeq ($(HOST), starlight)
  ARCH = nvidia
endif

### Apple (on MacBook laptops)
ifeq ($(ARCH),apple)
  CC = g++
  CFLAGS = -framework OpenCL    -DOPENCL_H
  LFLAGS = -framework OpenCL 
endif

### NVIDIA SDK (on MacBook laptops)
ifeq ($(ARCH),nvidia-laptop)
  CC = g++
  NVIDIA_SDK = /Developer/GPU\ Computing/

  CFLAGS = -W -Wall  -Wimplicit -Wswitch -Wformat  -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function  -fno-strict-aliasing   -I. -I$(NVIDIA_SDK)/OpenCL/common//inc -I$(NVIDIA_SDK)/shared/inc -DUNIX -DMAC -DNVIDIA -O3  

  LFLAGS =  -fPIC  -L$(NVIDIA_SDK)/OpenCL/common/lib -L$(NVIDIA_SDK)/shared/lib/ -L$(NVIDIA_SDK)/shared/lib/darwin  -framework OpenCL -framework OpenGL  -framework AppKit  
endif

### NVIDIA (gpu285, starlight, in Lille)
ifeq ($(ARCH),nvidia)
  CC = g++
  #NVIDIA_SDK = /usr/local/NVIDIA_GPU_SDK
  NVIDIA_SDK = /usr/local/cuda
  CFLAGS = -W -Wall -Wno-char-subscripts -Wimplicit -Wswitch -Wformat  -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function  -fno-strict-aliasing -I$(NVIDIA_SDK)/include  -I. -I$(NVIDIA_SDK)/OpenCL/common/inc -I$(NVIDIA_SDK)/shared/inc -DUNIX -DMAC -DNVIDIA -O3  
  LFLAGS = -lm -fPIC -L/usr/lib64/ -L$(NVIDIA_SDK)/OpenCL/common/lib -L$(NVIDIA_SDK)/shared/lib/ -L$(NVIDIA_SDK)/shared/lib/linux  -lOpenCL  
endif

### ATI (gpu4890, in Lille)
ifeq ($(ARCH),ati)
  CC = g++
  ATI_SDK = /usr/local/ati/
  CFLAGS =  -Wfloat-equal -Wpointer-arith  -g3    -I $(ATI_SDK)/include     -DATI
  LFLAGS =   -lpthread -ldl -L/usr/X11R6/lib   -lOpenCL    -L$(ATI_SDK)/lib/x86_64  # -lSDKUtil 
endif


# CC = g++  # some C++ in AMD code (FIXME)

.PHONY: exe clean info

exe: info
	$(CC) $(CFLAGS) $(FLAGS) -c $(TARGET).cpp
	$(CC) $(LFLAGS) $(FLAGS) $(TARGET).o -o $(TARGET).$(ARCH)


clean:
	rm -f $(TARGET).o
	rm -f $(TARGET).$(ARCH)
	rm -f $(TARGET).kernel.c.bin

info:
	@echo "HOST: " $(HOST)
	@echo "ARCH: " $(ARCH)
