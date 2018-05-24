# A makefile is direct complie tools

BUILD_DIR = build
#SOURCE_DIR = $(realpath ../test)
SOURCE_DIR = .

PYTHON ?= python3
PYTHON_CONFIG ?= python3-config

CXX = g++-7
CC = gcc-7

OPTFLAGS ?= -O3 -march=native

CXXFLAGS ?= -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
CXXFLAGS += $(DEFINES) $(OPTFLAGS) -fPIC
LDFLAGS ?= $(OPTFLAGS)
#LDFLAGS += -shared -fPIC


TF_CXXFLAGS ?= $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS ?= $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')


PYTHON_CXXFLAGS ?= $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LDFLAGS ?= $(shell $(PYTHON_CONFIG) --ldflags)
CXXFLAGS += -I. $(PYTHON_CXXFLAGS)
LDFLAGS  += -L. $(PYTHON_LDFLAGS)


## see pkg-config --cflags --libs $SEASTAR/build/release/seastar.pc
LIBS = seastar
LIBS_PATH = /home/tanguofu/workspace/AutoDL/shouter/seastar/build/debug

LIBS_CXXFLAGS = $(shell pkg-config --cflags $(LIBS_PATH)/$(LIBS).pc)
LIBS_LDFLAGS = $(shell pkg-config --libs $(LIBS_PATH)/$(LIBS).pc)



#CXXFLAGS += -I. $(TF_CXXFLAGS) $(PYTHON_CXXFLAGS) $(LIBS_CXXFLAGS)
#LDFLAGS  += -L. $(TF_LDFLAGS) $(PYTHON_LDFLAGS) $(LIBS_LDFLAGS)

CXXFLAGS += -I. $(LIBS_CXXFLAGS)
LDFLAGS  += -L. $(LIBS_LDFLAGS)


SHELL = bash

CC_SOURCES = $(shell find $(SOURCE_DIR) -name "*.cc" | sed 's/^\.\///g')
OBJS = $(addprefix $(BUILD_DIR)/,$(CC_SOURCES:.cc=.o))
TRAGETS = $(basename $(OBJS))

#OBJS = $(addprefix $(BUILD_DIR)/,$(CC_SOURCES:.cc=.o))
DEPFILES = $(OBJS:.o=.d)

EXT_SUFFIX ?= $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

TF_SO = $(BUILD_DIR)/zmq_pull_op$(EXT_SUFFIX)
PYBIND_SO = $(BUILD_DIR)/example$(EXT_SUFFIX)
CXXFLAGS += $(shell $(PYTHON) -c 'import pybind11; print("-I" + pybind11.get_include())')

#ALL_SO = $(TF_SO) $(PYBIND_SO)



.PHONY: all 
#all: $(ALL_SO)

#$(TF_SO): $(SOURCE_DIR)/%$(EXT_SUFFIX): $(BUILD_DIR)/%.o
#	@echo "Linking $@ ..."
#	@$(CXX) $^ -o $@ $(LDFLAGS) $(TF_LDFLAGS)
#	@echo "done."

all:$(PYBIND_SO) $(TRAGETS)

$(PYBIND_SO): $(SOURCE_DIR)/%$(EXT_SUFFIX): %.o
	@echo "Linking $@ ..."
	@$(CXX) -O3 -Wall -shared -std=c++11 -fPIC -I../pybind11/include `python3 -m pybind11 --includes` build/example.o -o $(PYBIND_SO)
	@echo "done."

build/%: build/%.o build/%.d 
	@echo "Linking $(CXX) $^ -o $@ $(LDFLAGS) $(PYTHON_LDFLAGS)"
	@$(CXX) $< -o $@ $(CXXFLAGS) $(LDFLAGS) 
	@echo "done."

build/%.o: %.cc
	@echo "$(CXX) -c $< -o $@ $(CXXFLAGS)"
	@$(CXX) -c $< -o $@ $(CXXFLAGS)

build/%.d: %.cc Makefile
	@echo "[dep] $< ..."
	@$(CXX) $(CXXFLAGS) -MM -MT "$(OBJ_DIR)/$(<:.cc=.o) $(OBJ_DIR)/$(<:.cc=.d)" "$<"  > "$@" || rm "$@"
	
clean:
	@rm -rvf $(BUILD_DIR)/*

