#For Intel Linux 32-bit
#ARCH=LINUXINTEL32

#For Windows 32/64-bit
#ARCH=WINDOWS

#For Intel Mac 32-bit, Maple IP:10.60.0.2 machine.
#ARCH=MAC32

#For Sparc Solaris 8 64-bit, Maple IP:10.60.0.4 machine.
#ARCH=SOLARIS64

#For PPC 64-bit, Maple IP:10.60.0.6 machine
#ARCH=PPC64

#For ADM 64-bit, Maple IP:10.60.0.8 machine
ARCH=LINUXINTEL64

#For MAC 64-bit, Maple IP:10.60.0.9 machine
#ARCH=MAC64

ifeq ($(ARCH), LINUXINTEL32)
#	CC = g++
	CC = gcc
	CFLAGS=-c  -O6 -fPIC -DLINUXINTEL32=1 
	SOFLAGS= -shared 
	LDFLAGS=-lm 
endif

ifeq ($(ARCH), WINDOWS)
	CC = cl
	LINK = link
	CFLAGS = /nologo /w /O2 /MD 
	LIBFLAGS = /DLL /nologo /Gz
endif

ifeq ($(ARCH), MAC32)
	CC = gcc
	CFLAGS= -O3 -fomit-frame-pointer -fno-common -fPIC -arch ppc -arch i386 -arch ppc64 -isysroot /Developer/SDKs/MacOSX10.4u.sdk -DMAC32=1
	SOFLAGS= -bundle -fPIC -arch ppc -arch i386  -arch ppc64
	LDFLAGS=-l
endif

ifeq ($(ARCH), SOLARIS64)
	CC = gcc 
	CFLAGS=-c  -O6 -fPIC -DSOLARIS64=1
	SOFLAGS= -shared
	LDFLAGS=-lm
endif

ifeq ($(ARCH), PPC64)
	CC = gcc
	CFLAGS= -O3 -fomit-frame-pointer -fno-common -fPIC -arch ppc -arch i386 -arch ppc64 -isysroot /Developer/SDKs/MacOSX10.4u.sdk -DPPC64=1
	SOFLAGS= -bundle -fPIC -arch ppc -arch ppc64
	LDFLAGS=-lm
endif

ifeq ($(ARCH), LINUXINTEL64)
	CC = gcc 
##CFLAGS=-c  -O6 -fPIC -DLINUXINTEL64=1 -pg
	CFLAGS=-c  -O6 -fPIC -DLINUXINTEL64=1
	SOFLAGS= -shared 
##LDFLAGS=-lm
	LDFLAGS=-lm -L. 
endif

ifeq ($(ARCH), MAC64)
	CC = gcc
	CFLAGS= -O3 -fomit-frame-pointer -fno-common -fPIC -arch ppc -arch i386 -arch ppc64 -isysroot /Developer/SDKs/MacOSX10.4u.sdk -DMAC64=1
	SOFLAGS= -bundle -fPIC -arch ppc -arch ppc64
	LDFLAGS=-lm
endif


SRC=generalFuncs.c  FMUL.c FDIV.c GCD.c HGCD.c matrix.c MultiDFFT.c MPMMTS.c \
	HashTable.c UniHensel.c FINTERP.c Factorization.c SubResultantSeq.c  \
	IteratedResultant.c LinkedList.c IsInvertible.c \
	RegularGcd.c CONSTANTS.c MapleCConverter.c solve2.c

ifeq ($(ARCH), WINDOWS)
OBJ=$(SRC:.c=.obj)
OBJEXE=$(SRCEXE:.c=.obj)
else
OBJ=$(SRC:.c=.o)
OBJEXE=$(SRCEXE:.c=.o)
endif

SRCEXE=$(SRC) TEST.c
EXEFILE=TEST


ifeq ($(ARCH), LINUXINTEL32)
soLib=libmodpn.so
#soLib=LibFastPoly.so.intel
endif

ifeq ($(ARCH), MAC32)
soLib=libmodpn.so
#soLib=LibFastPoly.so.02
endif
ifeq ($(ARCH), SOLARIS64)
soLib=libmodpn.so
#soLib=LibFastPoly.so.04
endif
ifeq ($(ARCH), PPC64)
soLib=libmodpn.so
#soLib=LibFastPoly.so.06
endif
ifeq ($(ARCH), LINUXINTEL64)
soLib=libmodpn.so
#soLib=LibFastPoly.so.08
endif
ifeq ($(ARCH), MAC64)
soLib=libmodpn.so
#soLib=LibFastPoly.so.09
endif
ifeq ($(ARCH), WINDOWS)
dllLib=modpn.dll
#dllLib=LibFastPoly.WIN64.dll
endif


ifeq ($(ARCH), WINDOWS)
all:maincc exe

%.obj:  $(SRC)
	$(CC) $(CFLAGS) /c $*.c
maincc: $(OBJ)
	$(LINK) $(LIBFLAGS) /out:$(dllLib) $(OBJ)
	mt /manifest modpn.dll.manifest /outputresource:modpn.dll\;#2;
	#mt /manifest LibFastPoly.WIN64.dll.manifest /outputresource:LibFastPoly.WIN64.dll\;#2;
#	cp ./LibFastPoly.WIN64.dll ../modpn.mpl/LibFastPoly.dll
#	cp ./LibFastPoly.WIN64.dll ../Triade/tst/LibFastPoly.dll

%.obj: $(SRCEXE)
	$(CC) $(CFLAGS) /c $*.c

exe: $(OBJEXE)
	$(LINK) /out:$(EXEFILE) $(OBJEXE)
else

maincc: $(OBJ)
	#creating shared library.
	$(CC)  $(SOFLAGS) -o $(soLib) $(OBJ)  $(LDFLAGS)
	#creating TEST file.
	$(CC)  $(CFLAGS) TEST.c 
	$(CC) $(OBJEXE) -o $(EXEFILE) $(LDFLAGS)
endif

doc:
	perl gdoc -html AS.h > AS.html
	perl gdoc -html inlineFuncs.h > inlineFuncs.html
	perl gdoc -html generalFuncs.c > generalFuncs.html
	perl gdoc -html FMUL.c > FMUL.html
	perl gdoc -html FDIV.c > FDIV.html
	perl gdoc -html GCD.c > GCD.html
	perl gdoc -html HGCD.c > HGCD.html
	perl gdoc -html matrix.c > matrix.html
	perl gdoc -html MultiDFFT.c > MultiDFFT.html
	perl gdoc -html MPMMTS.c > MPMMTS.html
	perl gdoc -html HashTable.c > HashTable.html
	perl gdoc -html UniHensel.c > UniHensel.html
	perl gdoc -html MapleCConverter.c > MapleCConverter.html
	perl gdoc -html FINTERP.c > FINTERP.html
	perl gdoc -html Factorization.c > Factorization.html
	perl gdoc -html SubResultantSeq.c >  SubResultantSeq.html
	perl gdoc -html IteratedResultant.c > IteratedResultant.html
	perl gdoc -html LinkedList.c > LinkedList.html
	perl gdoc -html IsInvertible.c > IsInvertible.html
	perl gdoc -html RegularGcd.c > RegularGcd.html
	perl gdoc -html CONSTANTS.c > CONSTANTS.html


clean:
	rm -f main *~ *.o *.op *.DAT  *.ini a.out mk test *.obj
	rm -f callgrind.out.* ex TEST  TEST-RESULT *.exp *.manifest
	rm -f \#*  TEST.lib
	rm -f *~
	rm -f libmodpn.*
