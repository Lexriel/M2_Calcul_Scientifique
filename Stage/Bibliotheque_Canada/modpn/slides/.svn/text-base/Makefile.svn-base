# Makefile
# MMM
# July 2011

SHELL = bash

# All files that require building
SRCS =  JiaJian_project
WIP = JiaJian_project

DVITARGS   = $(SRCS:%=%.dvi)
PSTARGS    = $(SRCS:%=%.ps)
PDFTARGS    = $(SRCS:%=%.pdf)
CLEANTARGS =  $(SRCS:%=%.clean)

wip: $(WIP).clean $(WIP).pdf

all: clean pdf

slides: clean pdf 

clean: cleanfiles

cleanfiles:
	-@rm -f *.aux *.log *.dvi *.ps *.pdf  *.idx *~ *.blg *.bbl *.bak *.out *.snm *.toc *.nav *.vrb

ps: dvi $(PSTARGS)

dvi: $(DVITARGS)

pdf: $(PDFTARGS)

%.clean:
	rm -f `basename $*`.pdf `basename $*`.ps `basename $*`.dvi `basename $*`.aux `basename $*`.log

%.pdf:
	pdflatex `basename $*`.tex ;
	pdflatex `basename $*`.tex ;
#	latex `basename $*`.tex ;
#	latex `basename $*`.tex ;
#	dvipdf `basename $*` ;


%.dvi: 
	latex `basename $*`.tex ;
	latex `basename $*`.tex ;
%.ps:
	-@(dvips -t landscape -P cmz -o  `basename $*`.ps `basename $*`.dvi >/dev/null ;)


handouts: $(WIP).pdf
	pdfnup --nup  2x2 --orient landscape $(WIP).pdf
##	-@(makehandouts $(WIP))

