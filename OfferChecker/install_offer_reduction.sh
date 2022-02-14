src=/home/andrea/Scrivania/Hotspot/OfferChecker/

g++  -c -fopenmp -fPIC ${src}offer_reduction_parallel.cc -o ${src}offer_parallel.o
g++ -shared -fopenmp -Wl,-soname,${src}liboffers_parallel.so -o ${src}liboffers_parallel.so ${src}offer_parallel.o