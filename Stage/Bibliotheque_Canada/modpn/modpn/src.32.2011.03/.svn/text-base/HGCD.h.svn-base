/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#ifndef __HGCD_h
#define __HGCD_h 

#include "Types.h"
#include "FMUL.h"
#include "FDIV.h"
#include "GCD.h"
#include "generalFuncs.h"
#include <time.h>

void XGCD(sfixn *C, sfixn *dC, sfixn *D, sfixn *dD, sfixn *G, sfixn *dG, 
        sfixn *A, sfixn dA, sfixn *B, sfixn dB, MONTP_OPT2_AS_GENE *pPtr);

sfixn* HalfGCD(sfixn *dG, sfixn *A, sfixn dA, sfixn *B, sfixn dB, 
        MONTP_OPT2_AS_GENE *pPtr);

void EX_QUO_Uni_Wrapper(sfixn *Q, sfixn *dQ, sfixn *A, sfixn dA, sfixn *B, 
        sfixn dB, MONTP_OPT2_AS_GENE *pPtr);

sfixn* EX_GCD_Uni_Wrapper(sfixn *dG, sfixn *A, sfixn dA, sfixn *B, sfixn dB, 
        MONTP_OPT2_AS_GENE *pPtr);

#endif
