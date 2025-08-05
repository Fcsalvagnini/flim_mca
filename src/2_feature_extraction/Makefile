LIBFLIM_MCA = /workdir/lib_flim_mca
LIBIFT = $(LIBFLIM_MCA)/lib
FLAGS = -fPIC -std=gnu11 -Wall -Wno-unused-result -pedantic
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

EXTERNALS_LD = -lm

OS=$(shell uname -s)

ifeq ($(OS), Darwin)
	EXTERNALS_LD += -Xpreprocessor -fopenmp -lomp
else
	EXTERNALS_LD += -fopenmp
endif

INCLUDES = $(LIBIFT_INC) $(LIBSVM_INC) $(LIBCBLAS_INC) $(LIBNIFTI_INC) $(LIBJPEG_INC) $(TSNE_INC) $(ZLIB_INC)
LIBS	 = $(LIBIFT_LD) $(LIBSVM_LD) $(LIBCBLAS_LD) $(EXTERNALS_LD)

ifeq ($(IFT_DEBUG), 1)
	export FLAGS += -pg -g -fsanitize=address -fsanitize=leak -DIFT_DEBUG=1
else
	export FLAGS += -O3
endif

#CUDA path in case IFT_GPU is enabled
export CUDA_DIR1=/usr/local/cuda
export CUDA_DIR2=/opt/cuda

ifeq ($(IFT_GPU), 1)
	export FLAGS += -DIFT_GPU=1
	INCLUDES += -I $(CUDA_DIR1)/include
	INCLUDES += -I $(CUDA_DIR2)/include
	LIBS	 += -L $(CUDA_DIR1)/lib64 -L $(CUDA_DIR2)/lib64 -lcublas -lcudart
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

%: %.c libift
	$(CC) $(FLAGS) $< -o $(BIN)/$@ $(INCLUDES) $(LIBS)

clean:
	rm -rf $(BIN)/*

# Clean everything including dependencies
distclean: clean clean-deps

.PHONY: setup-deps clean-deps refresh-deps libift clean distclean