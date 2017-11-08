CC = gcc # compiler used
CFLAGS = -ansi -Wall -pedantic -c # option for the object compilation
LDFLAGS = # options for the link edition
DEBUGFLAGS = -c -g # options for debugging
EXEC = tarot1.out french_tarot2.out # list of the executive files
DBGFILES = tarot1.db french_tarot2.db # list of debugging files

# to generate all executive files
all: $(EXEC)

# to generate all the executive files with information for gdb
alldbg: $(DBGFILES)

# rule to construct the executive files
%.out: %.o
	$(CC) -o $@ $^ $(LDFLAGS)

# rule to construct the debugging files
%.db: %.do
	$(CC) -o $@ $^ $(LDFLAGS)

# rule to construct object files
%.o: %.c
	$(CC) -o $@ $(CFLAGS) $<

# rule to construct debugging object files
%.do: %.c
	$(CC) -o $@ $(DEBUGFLAGS) $<

# dependencies for the executive files
french_tarot2.out: french_tarot2.o french_tarot1.o
tarot1.out: tarot1.o french_tarot1.o tarot0.o

# dependencies for the object files
french_tarot2.o: french_tarot2.c french_tarot1.h
french_tarot1.o: french_tarot1.c french_tarot1.h
tarot1.o: tarot1.c french_tarot1.h tarot0.h
tarot0.o: tarot0.c tarot0.h french_tarot1.h

# dependencies for the debugging files
french_tarot2.db: french_tarot2.do french_tarot1.do
tarot1.db: tarot1.do tarot0.do french_tarot1.do

# dependencies for the debugging object files
french_tarot2.do: french_tarot2.c 
french_tarot1.do: french_tarot1.c french_tarot1.h
tarot1.do: tarot1.c tarot0.h french_tarot1.h
tarot0.do: tarot0.c french_tarot1.h

# clean functions
clean:
	rm -f *.o *.do

mrproper: clean
	rm -f $(EXEC) $(DBGFILES)