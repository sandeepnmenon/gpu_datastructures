all: driver

driver: driver.cu basic_hashmap.cu
	nvcc --std=c++11 -o driver driver.cu basic_hashmap.cu -I.

clean:
	rm -f driver
