CC = gcc # compiler used
CFLAGS = -ansi -Wall -pedantic -c -Wmissing-prototypes # option for the object compilation
LDFLAGS = # options for the link edition
DEBUGFLAGS = -c -g # options for debugging
EXEC = tarot.out # list of the executive files
DBGFILES = tarot.db # list of debugging files

usage:
	@echo 'How to use make for these programs:'
	@echo ''
	@echo 'make all -> generates the files:' $(EXEC)
	@echo 'make allgdb -> generates the files:' $(DBGFILES)
	@echo 'make file -> generates the corresponding file (can be a .out, .db, .o or .do).'
	@echo 'make clean -> removes all the objects files.'
	@echo 'make mrproper -> removes the object files and the files:' $(EXEC) $(DBGFILES)

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
tarot.out: main.o tarot.o

# dependencies for the object files
tarot.o: tarot.c tarot.h
main.o: main.c tarot.h

# dependencies for the debugging files
tarot.db: main.do tarot.do

# dependencies for the debugging object files
tarot.do: tarot.c tarot.h
main.do: main.c tarot.h

# clean functions
clean:
	rm -f *.o *.do

mrproper: clean
	rm -f $(EXEC) $(DBGFILES)