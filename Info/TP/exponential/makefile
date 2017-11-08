CC = gcc # compiler used
CFLAGS = -ansi -Wall -pedantic -c # option for the object compilation
LDFLAGS = # options for the link edition
DEBUGFLAGS = -c -g # options for debugging
EXEC = menu.out time1.out time2.out # list of the executive files
DBGFILES = menu.db time1.db time2.db # list of debugging files
PLOTS = time1.ps time2.ps timeall.ps # the plot timings we want to create 
PLOT = gnuplot # what we use to create the plots

# to generate all executive files
all: $(EXEC)

# to generate the plots
plots: $(PLOTS)

# to generate all the executive files with information for gdb
alldbg: $(DBGFILES)

# rule to construct the executive files
%.out: %.o
	$(CC) -o $@ $^ $(LDFLAGS)

# rule to construct the plot files
%.ps: %.txt
	$(PLOT) <$<

# rule to construct the debugging files
%.db: %.do
	$(CC) -o $@ $^ $(LDFLAGS)

# rule to construct object files
%.o: %.c
	$(CC) -o $@ $(CFLAGS) $<

# rule to construc the .dat files
%.dat: %.out
	./$< >$@

# rule to construct debugging object files
%.do: %.c
	$(CC) -o $@ $(DEBUGFLAGS) $<

# dependencies for the executive files
menu.out: menu.o fact.o power.o exp_basic.o exp_better.o
time1.out: time1.o fact.o power.o exp_basic.o
time2.out: time2.o exp_better.o

# dependencies for the object files
menu.o: menu.c menu.h fact.h power.h exp_basic.h exp_better.h
time1.o: time1.c time1.h exp_basic.h
time2.o: time2.c time2.h exp_better.h
exp_basic.o: exp_basic.c exp_basic.h fact.h power.h
exp_better.o: exp_better.c exp_better.h
fact.o: fact.c fact.h
power.o: power.c power.h

# dependencies for the plotting files:
time1.ps: time1.txt time1.dat
time2.ps: time2.txt time2.dat
timeall.ps: timeall.txt time1.dat time2.dat

# dependencies for the .dat
time1.dat: time1.out
time2.dat: time2.out

# dependencies for the debugging files
menu.db: menu.do fact.do power.do exp_basic.do exp_better.do
time1.db: time1.do fact.do power.do exp_basic.do
time2.db: time2.do exp_better.do

# dependencies for the debugging object files
menu.do: menu.c menu.h fact.h power.h exp_basic.h exp_better.h
time1.do: time1.c time1.h exp_basic.h
time2.do: time2.c time2.h exp_better.h
exp_basic.do: exp_basic.c exp_basic.h fact.h power.h
exp_better.do: exp_better.c exp_better.h
fact.do: fact.c fact.h
power.do: power.c power.h

# clean functions
clean:
	rm *.o *.do *.dat

mrproper: clean
	rm $(EXEC) $(DBGFILES) $(PLOTS)