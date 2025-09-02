# Makefile for BFS executable

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Iinclude

# Source files
SRC = bfs.c mmio.c
OBJ = $(SRC:.c=.o)

# Output executable
TARGET = bfs bfs-level-sync-cuda bfs-level-sync-bitmap-cuda bfs-warp-centric

# Default target
all: bfs-level-sync-cuda bfs bfs-level-sync-bitmap-cuda bfs-warp-centric

bfs-level-sync-cuda: bfs-level-sync-cuda.cu mmio.c
	nvcc  -O3 -I./include/ -x c mmio.c -x cu bfs-level-sync-cuda.cu -o bfs-level-sync-cuda -Xcudafe "--diag_suppress=2464 --diag_suppress=550"

bfs-level-sync-bitmap-cuda: bfs-level-sync-bitmap-cuda.cu mmio.c
	nvcc  -O3 -I./include/ -x c mmio.c -x cu bfs-level-sync-bitmap-cuda.cu -o bfs-level-sync-bitmap-cuda -Xcudafe "--diag_suppress=2464 --diag_suppress=550"

bfs-warp-centric: bfs-warp-centric.cu mmio.c
	nvcc  -O3 -I./include/ -x c mmio.c -x cu bfs-warp-centric.cu -o bfs-warp-centric -Xcudafe "--diag_suppress=2464 --diag_suppress=550"

# Link object files to create executable
bfs: $(OBJ)
	$(CC)  $(CFLAGS) -o $@ $^

# Compile .c files to .o
%.o: %.c
	$(CC) -DTESTING $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
