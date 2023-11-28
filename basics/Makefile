# Variables
EXECUTABLE = driver
SOURCE = driver.cu
FLAGS = -I. -g -G
NVCC = nvcc

# Default target
all: $(EXECUTABLE)

# Build the driver executable
$(EXECUTABLE): $(SOURCE)
	$(NVCC) --std=c++17 $(FLAGS) -o $(EXECUTABLE) $(SOURCE)

# Clean up the build
clean:
	rm -f $(EXECUTABLE)

# Run the executable
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# PHONY targets
.PHONY: all clean run
