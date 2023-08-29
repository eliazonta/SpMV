CC := gcc
NVCC := /usr/local/cuda-12.2/bin/nvcc
BIN_FOLDER := bin
SRC_FOLDER := src
PARALLEL_FOLDER := parallel
INCLUDE_FOLDER := include
NN-SEQ := SpMV-SEQ
NN-CUDA := SpMV-CUDA
SRC-SEQ := main.c
SRC-CUDA := main.cu

# =======================================

all: sequential cuda


sequential:
	@mkdir -p $(BIN_FOLDER)
	$(CC)  $(SRC_FOLDER)/$(SRC-SEQ) $(SRC_FOLDER)/parser.c $(SRC_FOLDER)/sequential.c
	@mv a.out $(BIN_FOLDER)/$(NN-SEQ)

parallel:
	@mkdir -p $(BIN_FOLDER)
	$(NVCC) $(SRC_FOLDER)/$(PARALLEL_FOLDER)/$(SRC-CUDA) $(SRC_FOLDER)/$(PARALLEL_FOLDER)/parallel.cu
	@mv a.out $(BIN_FOLDER)/$(NN-CUDA)
	
clean:
	rm -rf $(BIN_FOLDER) prova 
