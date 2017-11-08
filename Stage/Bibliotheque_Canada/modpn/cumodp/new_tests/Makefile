SHELL = bash

# Files location
TOPDIR = $(PWD)
SRCS = Taylor_shift

TESTTARGS = $(SRCS:%=%.test)
CLEANTARGS = $(SRCS:%=%.clean)



tests: $(TESTTARGS)

clean: $(CLEANTARGS)

%.test:
	(cd $(TOPDIR)/$* ; make)

%.clean:
	(cd $(TOPDIR)/$* ; make clean)
