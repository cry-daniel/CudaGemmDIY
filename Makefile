
#########################################################################
#
# Makefile for cuda_test
#
#########################################################################

TEST_SOURCE = src/test.cu

BIN_DIR := ./bin

TARGETBIN := $(BIN_DIR)/test

CC = nvcc

$(TARGETBIN):$(TEST_SOURCE) src/kernel.cu
	mkdir -p $(BIN_DIR)
	$(CC) $(TEST_SOURCE) --compiler-options=-fopenmp -lcublas -o $(TARGETBIN)
	$(TARGETBIN)

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)