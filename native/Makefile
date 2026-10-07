# liblanesim.so: the native tick. No -ffast-math and no FMA contraction, so float32 results can match XLA's bit
# for bit where the operation order is the same.
CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -fPIC -fopenmp -ffp-contract=off -Wall -Wextra -Wno-unused-parameter
BUILD ?= build
SRC := $(wildcard src/*.cpp src/champ/*.cpp src/champ/*/*.cpp)
OBJ := $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(SRC))
HDR := $(wildcard src/*.hpp src/champ/*.hpp src/champ/*/*.hpp src/gen/*)

$(BUILD)/liblanesim.so: $(OBJ)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(OBJ)

$(BUILD)/obj/%.o: src/%.cpp $(HDR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD)

.PHONY: clean
