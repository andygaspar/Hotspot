g++  -c -fopenmp -fPIC offer_eval_parallel.cc -o offer_parallel.o
g++ -shared -fopenmp -Wl,-soname,liboffers_parallel.so -o liboffers_parallel.so offer_parallel.o

pip install numpy, pandas, xpress, scipy