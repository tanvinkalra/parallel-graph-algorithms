# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Iinclude

# Source files
SRC = dfs.c mmio.c
OBJ = $(SRC:.c=.o)

# Output executable
TARGET = dfs dfs-cuda

# Default target
all: dfs dfs-cuda

dfs-cuda: dfs-cuda.cu mmio.c
	nvcc  -O3 -I./include/ -x c mmio.c -x cu dfs-cuda.cu -o dfs-cuda -Xcudafe "--diag_suppress=2464 --diag_suppress=550"

# Link object files to create executable
dfs: $(OBJ)
	$(CC)  $(CFLAGS) -o $@ $^

# Compile .c files to .o
%.o: %.c
	$(CC) -DTESTING $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
