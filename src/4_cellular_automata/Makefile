LIBFLIM_MCA = /workdir/lib_flim_mca
LIBIFT = $(LIBFLIM_MCA)/lib
BIN = $(LIBFLIM_MCA)/bin

LIBSVM_DIR = $(LIBIFT)/externals/libsvm
LIBNIFTI_DIR= $(LIBIFT)/externals/libnifti
LIBJPEG_DIR= $(LIBIFT)/externals/libjpeg
LIBPNG_DIR= $(LIBIFT)/externals/libpng
TSNE_DIR= $(LIBIFT)/externals/tsne
ZLIB=$(LIBIFT)/externals/zlib

LIBIFT_INC   = -I $(LIBIFT)/include
LIBIFT_LD	= -L $(LIBIFT) -lift
LIBSVM_INC   = -I $(LIBSVM_DIR)/include
LIBSVM_LD	= -L $(LIBSVM_DIR)/lib -lstdc++
LIBCBLAS_INC = -I/usr/include/x86_64-linux-gnu
LIBCBLAS_LD  = -L/usr/lib/x86_64-linux-gnu -lopenblas

LIBNIFTI_INC = -I $(LIBNIFTI_DIR)/include
LIBJPEG_INC = -I $(LIBJPEG_DIR)/include
TSNE_INC = -I $(TSNE_DIR)/include
ZLIB_INC = -I $(ZLIB)/include

OS=$(shell uname -s)

# For Mac OS, add a special flag to enable the openmp. Install: brew install libomp
# Otherwise, add the openmp flag commonly
ifeq ($(OS), Darwin)
	EXTERNALS_LD += -Xpreprocessor -fopenmp -lomp
	OPENMP_FLAGS = -Xpreprocessor -fopenmp
	OPENMP_LINK = -lomp
else
	EXTERNALS_LD += -lm -fopenmp
	OPENMP_FLAGS = -fopenmp
	OPENMP_LINK = -lgomp
endif

INCLUDES = $(LIBIFT_INC) $(LIBSVM_INC) $(LIBCBLAS_INC) $(LIBNIFTI_INC) $(LIBJPEG_INC) $(TSNE_INC) $(ZLIB_INC)
LIBS	 = $(LIBIFT_LD) $(LIBSVM_LD) $(LIBCBLAS_LD) -lm

# Separate libs for NVCC (without direct OpenMP flags)
NVCC_LIBS_BASE = $(LIBIFT_LD) $(LIBSVM_LD) $(LIBCBLAS_LD) -lm

# Base flags for C compilation (without OpenMP)
BASE_FLAGS = -fPIC -Wall -Wno-unused-result -pedantic

ifeq ($(IFT_DEBUG), 1)
	export FLAGS += -pg -g -fsanitize=address -fsanitize=leak -DIFT_DEBUG=1 $(OPENMP_FLAGS)
	BASE_FLAGS += -pg -g -DIFT_DEBUG=1
else
	export FLAGS += -O3 $(OPENMP_FLAGS)
	BASE_FLAGS += -O3
endif

#CUDA path in case IFT_GPU is enabled
export CUDA_DIR1=/usr/local/cuda
export CUDA_DIR2=/opt/cuda

ifeq ($(IFT_GPU), 1)
	export FLAGS += -DIFT_GPU=1
	BASE_FLAGS += -DIFT_GPU=1
	INCLUDES += -I $(CUDA_DIR1)/include
	INCLUDES += -I $(CUDA_DIR2)/include
	LIBS	 += -L $(CUDA_DIR1)/lib64 -L $(CUDA_DIR2)/lib64 -lcublas -lcudart -lcudadevrt
	MAKE_CMD=$(MAKE) IFT_GPU=1

	# CUDA compiler settings
	NVCC = nvcc
	NVCC_FLAGS = -std=c++11 -O3 -arch=sm_50

	# Pass host compiler flags properly to NVCC (excluding OpenMP)
	HOST_COMPILER_FLAGS = $(BASE_FLAGS)
	NVCC_FLAGS += $(foreach flag,$(HOST_COMPILER_FLAGS),-Xcompiler $(flag))

	# CUDA libraries
	CUDA_LIBS = -L $(CUDA_DIR1)/lib64 -L $(CUDA_DIR2)/lib64 -lcublas -lcudart -lcudadevrt

	# Complete NVCC libs (base libs + CUDA libs + OpenMP for linking)
	ifeq ($(OS), Darwin)
		NVCC_LIBS_COMPLETE = $(NVCC_LIBS_BASE) $(CUDA_LIBS) -Xlinker -lomp
	else
		NVCC_LIBS_COMPLETE = $(NVCC_LIBS_BASE) $(CUDA_LIBS) -lgomp
	endif
else
	MAKE_CMD=$(MAKE)
endif

# Clean dependencies - removes lib directory contents
clean-deps:
	@echo "Cleaning lib directory..."
	@rm -rf $(LIBIFT)
	@mkdir -p $(LIBIFT)

# Extract dependencies and setup directories (with forced cleanup)
setup-deps: clean-deps
	@echo "Setting up dependencies..."
	@mkdir -p $(BIN)
	@if [ -f "$(LIBFLIM_MCA)/externals.tar.gz" ]; then \
		echo "Extracting externals.tar.gz..."; \
		cd $(LIBFLIM_MCA) && tar -xzf externals.tar.gz -C lib/; \
	fi
	@if [ "$(IFT_GPU)" = "1" ]; then \
		if [ -f "$(LIBFLIM_MCA)/lib_gpu.tar.gz" ]; then \
			echo "Extracting lib_gpu.tar.gz for GPU build..."; \
			cd $(LIBFLIM_MCA) && tar -xzf lib_gpu.tar.gz; \
		else \
			echo "Warning: lib_gpu.tar.gz not found, falling back to CPU version"; \
			if [ -f "$(LIBFLIM_MCA)/lib_cpu.tar.gz" ]; then \
				echo "Extracting lib_cpu.tar.gz..."; \
				cd $(LIBFLIM_MCA) && tar -xzf lib_cpu.tar.gz; \
			fi; \
		fi; \
	else \
		if [ -f "$(LIBFLIM_MCA)/lib_cpu.tar.gz" ]; then \
			echo "Extracting lib_cpu.tar.gz for CPU build..."; \
			cd $(LIBFLIM_MCA) && tar -xzf lib_cpu.tar.gz; \
		else \
			echo "Error: lib_cpu.tar.gz not found"; \
			exit 1; \
		fi; \
	fi

# Force re-extraction of dependencies
refresh-deps: setup-deps

# Modified libift target - no need to build since libift.a exists
libift: setup-deps
	@echo "Using pre-compiled libift.a"
	@echo "Looking for libift.a in: $(LIBIFT)"
	@if [ ! -f "$(LIBIFT)/libift.a" ]; then \
		echo "Error: libift.a not found in $(LIBIFT)"; \
		echo "Contents of $(LIBIFT):"; \
		ls -la $(LIBIFT)/ 2>/dev/null || echo "Directory does not exist"; \
		echo "Contents of $(LIBFLIM_MCA):"; \
		ls -la $(LIBFLIM_MCA)/ 2>/dev/null || echo "Directory does not exist"; \
		exit 1; \
	fi
	@echo "Build configuration: $(if $(filter 1,$(IFT_GPU)),GPU,CPU) mode"

# Rule for regular C files
%: %.c libift
	$(CC) $(FLAGS) $< -o $(BIN)/$@ $(INCLUDES) $(LIBS)

# Special rule for GPU version
iftCAGpu: iftCAGpu.cu
	$(MAKE) libift
ifeq ($(IFT_GPU), 1)
	$(NVCC) $(NVCC_FLAGS) -x cu iftCAGpu.cu -o $(BIN)/iftCAGpu $(INCLUDES) $(NVCC_LIBS_COMPLETE)
else
	@echo "Error: GPU support is not enabled. Set IFT_GPU=1 in Makefile"
	@exit 1
endif

clean:
	rm -rf $(BIN)/*;

# Clean everything including dependencies
distclean: clean clean-deps

.PHONY: setup-deps clean-deps refresh-deps libift clean distclean