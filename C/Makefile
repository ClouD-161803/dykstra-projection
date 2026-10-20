# Builds the two shared libraries with gcc or clang (e.g. make CC=clang).
# The suffix is chosen for the platform; override it with e.g. make SO=.dll
# under MinGW.  -ffp-contract=off keeps a compiler from fusing a multiply and
# an add into one rounding, so that every platform runs the same arithmetic.
CC = cc
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
SO = .dylib
else
SO = .so
endif
CFLAGS = -std=c11 -O2 -Wall -Wextra -fPIC -ffp-contract=off
LDLIBS = -lm

all: accelerated$(SO) dykstra$(SO)

accelerated$(SO): accelerated.c accelerated.h
	$(CC) $(CFLAGS) -shared accelerated.c -o accelerated$(SO) $(LDLIBS)

dykstra$(SO): dykstra.c dykstra.h
	$(CC) $(CFLAGS) -shared dykstra.c -o dykstra$(SO) $(LDLIBS)

clean:
	rm -f accelerated.so accelerated.dylib accelerated.dll dykstra.so dykstra.dylib dykstra.dll

.PHONY: all clean
