CC=g++
CUDAINCLUDE=${CONDA_PREFIX}/include
CUDALIB=${CONDA_PREFIX}/lib

libinttemp.so: utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp
	$(CC) -O3 -fPIC -shared utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread -o libinttemp.so

all:
	make libinttemp.so

clean:
	rm -rf *.o libinttemp.so
