all: driver

driver: driver.cu
	nvcc --std=c++17 -o driver driver.cu -I.

clean:
	rm -f driver
