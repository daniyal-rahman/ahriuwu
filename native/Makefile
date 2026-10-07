# liblanesim.so: the native lane-slice tick. No -ffast-math and no FMA contraction, so float32 results can match
# XLA's bit for bit where the operation order is the same.
CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -fPIC -fopenmp -ffp-contract=off -Wall -Wextra -Wno-unused-parameter
BUILD ?= build

$(BUILD)/liblanesim.so: src/api.cpp src/tick.cpp src/*.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -shared -o $@ src/api.cpp src/tick.cpp

clean:
	rm -rf $(BUILD)

.PHONY: clean
