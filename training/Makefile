CC = g++
CXXFLAGS := -I ./ -O2 -std=c++17 -Wall

markovSeq:  markovSeq.o
	$(CXX) $(CXXFLAGS) markovSeq.o -o markovSeq
markovSeq.o : markovSeq.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@
markovSeq.o: markov.h
clean:
	rm -f markovSeq markovSeq.o
