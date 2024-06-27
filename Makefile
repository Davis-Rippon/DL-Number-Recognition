GPP = g++
FLAGS = -g -Wall
SRC = src
OBJ = obj

SRCS = $(wildcard $(SRC)/*.cpp)
OBJS = $(patsubst $(SRC)/%.cpp, $(OBJ)/%.o, $(SRCS))

BINDIR = bin
BIN = bin/main

all:$(BIN)

$(BIN): $(OBJS)
	$(GPP) $(FLAGS) $(OBJS) -o $@

$(OBJ)/%.o: $(SRC)/%.cpp
	$(GPP) $(FLAGS) -c $< -o $@

clean:
	$(RM) -r $(BINDIR)/* $(OBJ)/*
