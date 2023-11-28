all: driver

driver: driver.cpp basic_hashmap.cu
	nvcc -o driver driver.cpp basic_hashmap.cu -I.

clean:
	rm -f driver
