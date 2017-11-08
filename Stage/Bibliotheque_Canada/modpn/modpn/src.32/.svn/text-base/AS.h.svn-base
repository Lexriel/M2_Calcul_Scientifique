/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#ifndef __AS_h
#define __AS_h 


#include "Types.h"
#include "CONSTANTS.h"


extern sfixn BASE;
extern sfixn BASE_1;
extern sfixn BASEHALF;



#ifdef LINUXINTEL32

/**
 * MulHiLoUnsigned: 
 * @h: pointer to a sfixn "a".
 * @l: pointer to a sfixn "b".
 *
 * Computing sfixn multiplication and return the high word and low word.
 *           I.e. *h = HighPart(ab), *l = LowPart(ab)
 * Return value: void.
 **/


// If Intel 32 Linux machine use this...

static inline void
MulHiLoUnsigned(sfixn *h, sfixn *l)
{
  __asm__ ("mull %3" : "=a" (*l), "=d" (* h) : "%0" (* h), "rm" (* l));
}



#else


// generic C code.


// the following signature has been modified by Xin 
//  On Nov.19.2008. to fix the g++ compilation (sfixn<->usfixn) problem.
static void
//MulHiLoUnsigned(usfixn *h, usfixn *l)
MulHiLoUnsigned(sfixn *h, sfixn *l)
{
  ulongfixnum prod;
  prod=(ulongfixnum)(usfixn)(*h) * (ulongfixnum)(usfixn)(*l);
  //  printf("a*b=%ld\n", prod);
  

/* hack to fix old GCC on Sparc 32-bit */
/* (big endian)  prod = [hi 32, lo 32] */
#if SOLARIS64
  *h=  *((usfixn *)&prod);
#else
  *h=  (sfixn)(((ulongfixnum)prod)>>BASE);
#endif
  *l=  (sfixn)(usfixn)prod;

  // printf("*h=%ld\n", *h);
  // printf("*l=%ld\n", *l);
}


#endif

#endif
