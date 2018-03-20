
CC = clang++ -I /usr/include/eigen3

CXXFLAGS  = -std=c++11 -Wall -fopenmp=libomp #-msse2

default: main

main:  main.o
	$(CC) $(CXXFLAGS) -o main main.o

main.o:  main.cpp BaseNode.h
	$(CC) $(CXXFLAGS) -c main.cpp

#BaseNode.o: Classifier.cpp Classifier.h Dataset_reader.h Neurons.h
#	$(CC) $(CXXFLAGS) -c Classifier.cpp

#FirstNode.o: Dataset_reader.cpp Dataset_reader.h
#	$(CC) $(CXXFLAGS) -c Dataset_reader.cpp

#InternalNode.o: Neurons.cpp Neurons.h
#	$(CC) $(CXXFLAGS) -c Neurons.cpp

clean:
	$(RM) count *.o *~
