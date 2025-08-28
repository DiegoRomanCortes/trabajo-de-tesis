all:
	gcc -static -O3 array1d.c -o  array1d -lfftw3_threads -lfftw3 -lgslcblas -lm -lpthread
	./array1d

clean:
	rm -f array1d