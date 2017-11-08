# Makefile 
# Marc Moreno Maza 
# Décembre 2002

SHELL = bash

SRCS = CPUtime_e CPUtime_n GPUtime_e GPUtime_n time_small time_big # plotGPUprime plotHORprime

PLOTARGS = $(SRCS:%=%.gnuplot)
CLEANTARGS = $(SRCS:%=%.clean)

all: clean gnuplot

gnuplot: $(PLOTARGS)

clean: cleanfiles
	rm -f *~  gnu/*~

cleanfiles: $(CLEANTARGS)

%.gnuplot:
	gnuplot < gnu/`basename $*`.gnu


%.clean:
	rm -f eps/`basename $*`.eps
