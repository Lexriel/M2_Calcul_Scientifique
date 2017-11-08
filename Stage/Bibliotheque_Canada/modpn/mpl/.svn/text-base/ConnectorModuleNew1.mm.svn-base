ConnectorModuleNew1 := module() 

local trampoline; 

export CrePtsPTrees, FastEvalM, GetResChainsTFT, GetResChains, FastInterpM,
    FastInterpithdthM, FastInterpithM, FastInterpNextLCM, DftEvalM, TFTEvalM,
    TFTInterpM, DftInterpM, DftInterpithdthM, TFTInterpithdthM, DftInterpithM,
    TFTInterpithM, DftInterpNextLCM, TFTInterpNextLCM, createGoodRoots,
    HenselLiftingC, IsInvertableCN, NormalFormCN, NormalizeTriSetCN,
    GetQuotientImage, ReduceCoefficientCN, QuoModTriSetCN, 
    FastInterpRatFunCons, IterResultantOneDimCN, IterResultantZeroDimCN,
    IsInvertibleZeroDimCN, FastInterpNextDefectiveLCM, DftInterpNextDefectiveLCM,
    TFTInterpNextDefectiveLCM, RegularGcdZeroDimCN, ResultantMultiCN,
    IsCudaEnabledCN, EnableCudaCN;

####################################
## exported ALGEB functions from C
####################################

export MapleGcArray, TFTEvalM_ALGEB, TFTInterpM_ALGEB, GetResChainsTFT_ALGEB,
    TFTInterpithdthM_ALGEB, TFTInterpithM_ALGEB, TFTInterpNextLCM_ALGEB, 
    TFTInterpNextDefectiveLCM_ALGEB, DftEvalM_ALGEB, DftInterpM_ALGEB,
    GetResChains_ALGEB, DftInterpithdthM_ALGEB, DftInterpithM_ALGEB,
    DftInterpNextDefectiveLCM_ALGEB, DftInterpNextLCM_ALGEB,
    FastEvalM_ALGEB, FastInterpM_ALGEB, FastInterpithM_ALGEB,
    FastInterpithdthM_ALGEB, FastInterpNextDefectiveLCM_ALGEB,
    FastInterpNextLCM_ALGEB, BivariateSolveCN_ALGEB; 

option package;  

trampoline := proc() 
    local oldargs;
    oldargs := [args];
    proc() local newargs, lib;
       newargs := oldargs;
       # determine correct name for dymamic library
       lib := rhs(newargs[-1]):
       newargs := subsop(-1=('LIB' = ExternalCalling:-ExternalLibraryName(lib)), newargs):
       # determine correct integer datatype
       if kernelopts(wordsize)=32 then
          newargs := subs('INT'='integer[4]', newargs);
       else # kernelopts(wordsize)=64
          newargs := subs('INT'='integer[8]', newargs);
       end if:
       # reassign yourself to the externally defined routine
       unprotect(procname):
       assign(procname, define_external(op(newargs))):
       protect(procname):
       # call the external routine
       procname(args)
    end proc;
end proc;

######################################################################
# CUDA control
######################################################################
IsCudaEnabledCN := trampoline('is_cuda_tag_enabled', 
    'RETURN'::(integer[4]),
    'LIB' = "modpn"):

EnableCudaCN := trampoline('enable_cuda_tag',
    'flag'::(integer[4]), 
    'RETURN'::(integer[4]),
    'LIB' = "modpn"):

######################################################################
# Specialized Solver
######################################################################
BivariateSolveCN_ALGEB := trampoline('bivariate_solve_ALGEB', 
    'MAPLE', 'LIB'="modpn");

######################################################################
## Subproduct tree based functions 
######################################################################

CrePtsPTrees := trampoline('PTRTREESCRECN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
    'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dims1'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dims2'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
    'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
    'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dgs1'::(ARRAY(1 .. 1, 'integer[4]')), 
    'p1Sz'::(integer[4]), 
    'p1buff'::(ARRAY(1 .. 'p1Sz', 'integer[4]')), 
    'dgs2'::(ARRAY(1 .. 1, 'integer[4]')), 
    'p2Sz'::(integer[4]), 
    'p2buff'::(ARRAY(1 .. 'p2Sz', 'integer[4]')),  
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"); 

FastEvalM := trampoline('FastEvalMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
    'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
    'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
    'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ESz'::(integer[4]), 
    'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
    'dgs'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'fSz'::(integer[4]), 
    'fbuff'::(ARRAY(1 .. 'fSz', 'integer[4]')), 
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"); 

GetQuotientImage := trampoline('GetQuotientCN', 
    'N'::(integer[4]),
    'dd'::(integer[4]), 
    'SSz'::(integer[4]), 
    'S'::(ARRAY(1 .. 'SSz','integer[4]')), 
    'Edims1'::(ARRAY(1 .. 'N'+1, 'integer[4]')),
    'E1Sz'::(integer[4]), 
    'E1'::(ARRAY(1 .. 'E1Sz', 'integer[4]')),
    'Edims2'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'E2Sz'::(integer[4]),
    'E2'::(ARRAY(1 .. 'E2Sz', 'integer[4]')), 
    'p'::(integer[4]), 
    'opt'::(integer[4]), 
    'LIB' = "modpn");

FastInterpM := trampoline('FastInterpMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
    'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
    'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
    'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ESz'::(integer[4]), 
    'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')),  
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"); 

FastInterpithdthM := trampoline('InterpIthDthMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
    'ith'::(integer[4]), 
    'dth'::(integer[4]), 
    'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
    'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
    'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'LIB' = "modpn"); 

FastInterpithM := trampoline('InterpIthMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
    'ith'::(integer[4]), 
    'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
    'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
    'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 1, 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 1, 'integer[4]')), 
    'LIB' = "modpn"): 

FastInterpNextLCM := trampoline('InterpNextLCMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
    'start'::(integer[4]), 
    'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
    'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
    'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')),  
    'RETURN'::(integer[4]),
    'LIB' = "modpn"): 

FastInterpNextDefectiveLCM := trampoline('InterpNextDefectiveLCMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
    'start'::(integer[4]), 
    'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
    'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
    'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')),  
    'RETURN'::(integer[4]),
    'LIB' = "modpn"): 

######################################################################
## FFT based functions for constructing SCUBE
######################################################################

DftEvalM := trampoline('DftMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'es'::(ARRAY(1 .. 1, 'integer[4]')),
    'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ESz'::(integer[4]), 
    'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
    'dgs'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'fSz'::(integer[4]), 
    'fbuff'::(ARRAY(1 .. 'fSz', 'integer[4]')), 
    'rootsPtr'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'LIB' = "modpn"); 

DftInterpM := trampoline('InvDftMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'es'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ESz'::(integer[4]), 
    'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'rootsPtr'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'LIB' = "modpn"); 

GetResChains := trampoline('SubResultantChains', 
    'N'::(integer[4]), 
    'w'::(integer[4]), 
    'SSz'::(integer[4]), 
    'S'::(ARRAY(1 .. 'SSz', 'integer[4]')), 
    'Edims1'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'E1Sz'::(integer[4]), 
    'E1'::(ARRAY(1 .. 'E1Sz', 'integer[4]')), 
    'Edims2'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'E2Sz'::(integer[4]), 
    'E2'::(ARRAY(1 .. 'E2Sz', 'integer[4]')), 
    'p'::(integer[4]),  
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"); 

DftInterpithdthM := trampoline('InvDftIthDthMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'ith'::(integer[4]), 
    'dth'::(integer[4]), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'LIB' = "modpn"); 

DftInterpithM := trampoline('InvDftIthMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'ith'::(integer[4]), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'LIB' = "modpn"): 

DftInterpNextDefectiveLCM := trampoline('InvDftNextDefectiveLCMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'start'::(integer[4]), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'RETURN'::(integer[4]),
    'LIB' = "modpn"):

DftInterpNextLCM := trampoline('InvDftNextLCMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'start'::(integer[4]), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'RETURN'::(integer[4]),
    'LIB' = "modpn"):

createGoodRoots := trampoline('createGoodRootsCN', 
    'N'::(integer[4]), 
    'M'::(integer[4]), 
    'f1dgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'f1Buffer'::(ARRAY(1 .. 1, 'integer[4]')), 
    'f2dgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'f2Buffer'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'es'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'dims'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'p'::(integer[4]), 
    'RETURN'::(integer[4]),
    'LIB' = "modpn"):

######################################################################
##  NormalForm and misc functions
######################################################################

HenselLiftingC := trampoline('NewtonLiftUniCN',  
    'outPDGVECS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'outCOEFVECS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Y'::(integer[4]), 
    'y0'::(integer[4]), 
    'N'::(integer[4]), 
    'GNS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'MDAS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'TS_DGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inDGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inSIZS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'inCOEFS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'p'::(integer[4]),  
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"):

IsInvertableCN := trampoline('isInvertableCN',  
    'N'::(integer[4]), 
    'fDgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fBuffer'::(ARRAY(1 .. 1, 'integer[4]')), 
    'TS_DGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inDGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inSIZS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'inCOEFS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'p'::(integer[4]),  
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"):

NormalFormCN := trampoline('MultiModCN', 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'N'::(integer[4]), 
    'fDgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fBuffer'::(ARRAY(1 .. 1, 'integer[4]')), 
    'TS_DGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inDGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inSIZS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'inCOEFS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'p'::(integer[4]),  
    'opt'::(integer[4]), 
    'RETURN'::(integer[4]),  
    'LIB' = "modpn"):

IterResultantZeroDimCN := trampoline('IterResZeroDimCN', 
    'M'::(integer[4]), 
    'fDgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fBuffer'::(ARRAY(1 .. 1, 'integer[4]')), 
    'N'::(integer[4]), 
    'TS_DGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inDGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inSIZS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'inCOEFS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'p'::(integer[4]), 
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"):

ReduceCoefficientCN := trampoline('ReduceCoeffCN', 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'N'::(integer[4]), 
    'fDgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fBuffer'::(ARRAY(1 .. 1, 'integer[4]')), 
    'TS_DGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inDGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inSIZS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'inCOEFS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'p'::(integer[4]),  
    'RETURN'::(integer[4]),   
    'LIB' = "modpn"):

QuoModTriSetCN:= trampoline('QuotientModTriSetCN', 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'N'::(integer[4]), 
    'fDgs1'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fBuffer1'::(ARRAY(1 .. 1, 'integer[4]')),  
    'fDgs2'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fBuffer2'::(ARRAY(1 .. 1, 'integer[4]')), 
    'TS_DGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inDGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inSIZS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'inCOEFS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'p'::(integer[4]),  
    'LIB' = "modpn"):

NormalizeTriSetCN := trampoline('NormalizeCN',  
    'outPDGVECS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'outCOEFVECS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'N'::(integer[4]), 
    'TS_DGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inDGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inSIZS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'inCOEFS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'p'::(integer[4]),  
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"):

ResultantMultiCN := trampoline('ResultantMultivariateCN',
    'Nnew'::(ARRAY(1 .. 1, 'integer[4]')), 
    'outPDGVECS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'outCOEFVECS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'N'::(integer[4]), 
    'polyDgs1'::(ARRAY(1 .. 1, 'integer[4]')), 
    'polyCoef1'::(ARRAY(1 .. 1, 'integer[4]')),
    'polyDgs2'::(ARRAY(1 .. 1, 'integer[4]')), 
    'polyCoef2'::(ARRAY(1 .. 1, 'integer[4]')), 
    'p'::(integer[4]),  
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"):

FastInterpRatFunCons := trampoline('FastInterpRFRWrapC', 
    'itemNop'::(ARRAY(1 .. 1, 'integer[4]')), 
    'InterpedPtsNum'::(ARRAY(1 .. 1, 'integer[4]')),  
    'InterpedPtsDen'::(ARRAY(1 .. 1, 'integer[4]')), 
    'EvaluatingPts'::(ARRAY(1 .. 1, 'integer[4]')), 
    'EvaluatedPts'::(ARRAY(1 .. 1, 'integer[4]')), 
    'h'::(integer[4]), 
    'W'::(ARRAY(1 .. 1, 'integer[4]')), 
    'NoNodes'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Bases'::(ARRAY(1 .. 1, 'integer[4]')), 
    'data'::(ARRAY(1 .. 1, 'integer[4]')),  
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"):

IterResultantOneDimCN := trampoline('IterResOneDimCN', 
    'outVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'M'::(integer[4]), 
    'fDgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fBuffer'::(ARRAY(1 .. 1, 'integer[4]')), 
    'N'::(integer[4]), 
    'TS_DGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inDGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inSIZS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'inCOEFS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'bound'::(integer[4]), 
    'freeVarNo'::(integer[4]),  
    'p'::(integer[4]), 
    'RETURN'::(integer[4]),  
    'LIB' = "modpn"):

RegularGcdZeroDimCN := trampoline('RegularGcdChainCN', 
    'Ns'::(ARRAY(1 .. 1, 'integer[4]')), 
    'polyDgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'polyCoef'::(ARRAY(1 .. 1, 'integer[4]')), 
    'trisetDgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'trisetCoef'::(ARRAY(1 .. 1, 'integer[4]')), 
    'N'::(integer[4]), 
    'M'::(integer[4]), 
    'fDgs1'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fBuffer1'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fDgs2'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fBuffer2'::(ARRAY(1 .. 1, 'integer[4]')), 
    'TS_DGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inDGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inSIZS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'inCOEFS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'p'::(integer[4]),  
    'RETURN'::(integer[4]),
    'LIB' = "modpn"):

IsInvertibleZeroDimCN := trampoline('IsInvertibleChainCN', 
    'Ns'::(ARRAY(1 .. 1, 'integer[4]')), 
    'polyDgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'polyCoef'::(ARRAY(1 .. 1, 'integer[4]')), 
    'trisetDgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'trisetCoef'::(ARRAY(1 .. 1, 'integer[4]')), 
    'N'::(integer[4]), 
    'fDgs'::(ARRAY(1 .. 1, 'integer[4]')), 
    'fBuffer'::(ARRAY(1 .. 1, 'integer[4]')), 
    'TS_DGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inDGS'::(ARRAY(1 .. 1, 'integer[4]')), 
    'inSIZS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'inCOEFS'::(ARRAY(1 .. 1, 'integer[4]')),  
    'p'::(integer[4]),  
    'RETURN'::(integer[4]),
    'LIB' = "modpn"):

###############################################################################
##  TFT based fucntions for constructing SCUBE, non-ALGEB version
###############################################################################

GetResChainsTFT := trampoline('SubResultantChainsTFT', 
    'N'::(integer[4]), 
    'M'::(integer[4]), 
    'w'::(integer[4]), 
    'SSz'::(integer[4]), 
    'S'::(ARRAY(1 .. 'SSz', 'integer[4]')), 
    'Edims1'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'E1Sz'::(integer[4]), 
    'E1'::(ARRAY(1 .. 'E1Sz', 'integer[4]')), 
    'Edims2'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'E2Sz'::(integer[4]), 
    'E2'::(ARRAY(1 .. 'E2Sz', 'integer[4]')), 
    'p'::(integer[4]),  
    'RETURN'::(integer[4]), 
    'LIB' = "modpn"); 

TFTEvalM := trampoline('TFTMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'es'::(ARRAY(1 .. 1, 'integer[4]')),
    'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ESz'::(integer[4]), 
    'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
    'dgs'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'fSz'::(integer[4]), 
    'fbuff'::(ARRAY(1 .. 'fSz', 'integer[4]')), 
    'rootsPtr'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'LIB' = "modpn"); 

TFTInterpM := trampoline('InvTFTMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'es'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'ESz'::(integer[4]), 
    'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'rootsPtr'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
    'LIB' = "modpn"); 

TFTInterpithdthM := trampoline('InvTFTIthDthMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'ith'::(integer[4]), 
    'dth'::(integer[4]), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'LIB' = "modpn"); 

TFTInterpithM := trampoline('InvTFTIthMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'ith'::(integer[4]), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'LIB' = "modpn"): 

TFTInterpNextLCM := trampoline('InvTFTNextLCMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'start'::(integer[4]), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'RETURN'::(integer[4]),
    'LIB' = "modpn"):

TFTInterpNextDefectiveLCM := trampoline('InvTFTNextDefectiveLCMultiWrapCN', 
    'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
    'start'::(integer[4]), 
    'w'::(integer[4]), 
    'slicesz'::(integer[4]), 
    'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
    'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
    'Ssz'::(integer[4]), 
    'S'::(ARRAY(1 .. 1, 'integer[4]')), 
    'dVsz'::(integer[4]), 
    'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'cVsz'::(integer[4]), 
    'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
    'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
    'RETURN'::(integer[4]),
    'LIB' = "modpn"):

################################################################################
################################################################################
################################################################################
################################################################################
##   Functions taking ALGEB parameters and returning an ALGEB
################################################################################
################################################################################
################################################################################
################################################################################
            
## (0) ##
####################################
## C array management
##
## (1) a := MapleGcArray("malloc", 10);
##  
## (2) MapleGcArray("free", a);
##
####################################
MapleGcArray := trampoline('MapleGcArray_ALGEB', 'MAPLE', 'LIB'="modpn");

## (1) ##
###########################################################
## 'N'::(integer[4]), 
## 'M'::(integer[4]), 
## 'w'::(integer[4]), 
## 'SSz'::(integer[4]), 
## 'S'::(ARRAY(1 .. 'SSz', 'integer[4]')), 
## 'Edims1'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
## 'E1Sz'::(integer[4]), 
## 'E1'::(ARRAY(1 .. 'E1Sz', 'integer[4]')), 
## 'Edims2'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
## 'E2Sz'::(integer[4]), 
## 'E2'::(ARRAY(1 .. 'E2Sz', 'integer[4]')), 
## 'p'::(integer[4]),  
## 'RETURN'::(integer[4]), 
###########################################################
GetResChainsTFT_ALGEB := trampoline(
        'SubResultantChainsTFT_ALGEB', 'MAPLE', 'LIB' = "modpn"); 

## (2) ##
#############################################################
## 'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
## 'es'::(ARRAY(1 .. 1, 'integer[4]')),
## 'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
## 'ESz'::(integer[4]), 
## 'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
## 'dgs'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
## 'fSz'::(integer[4]), 
## 'fbuff'::(ARRAY(1 .. 'fSz', 'integer[4]')), 
## 'rootsPtr'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
#############################################################
TFTEvalM_ALGEB := trampoline(
        'TFTMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"); 

## (3) ##
#############################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'es'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ESz'::(integer[4]), 
##  'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
##  'rootsPtr'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
#############################################################
TFTInterpM_ALGEB :=
    trampoline('InvTFTMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"); 

## (4) ##
#############################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'ith'::(integer[4]), 
##  'dth'::(integer[4]), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
##  'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
#############################################################
TFTInterpithdthM_ALGEB :=
    trampoline('InvTFTIthDthMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"); 

## (5) ##
#############################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'ith'::(integer[4]), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
##  'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
#############################################################
TFTInterpithM_ALGEB := trampoline(
    'InvTFTIthMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"): 

## (6) ##
#############################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'start'::(integer[4]), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
##  'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'RETURN'::(integer[4]),
#############################################################
TFTInterpNextLCM_ALGEB :=
    trampoline('InvTFTNextLCMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"):

## (7) ##
#############################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'start'::(integer[4]), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
##  'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'RETURN'::(integer[4]),
#############################################################
TFTInterpNextDefectiveLCM_ALGEB :=
    trampoline('InvTFTNextDefectiveLCMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"):



######################################################################
######################################################################
######################################################################
######################################################################
## FFT based functions for constructing SCUBE, ALGEB version
######################################################################
######################################################################
######################################################################
######################################################################


########################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'es'::(ARRAY(1 .. 1, 'integer[4]')),
##  'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ESz'::(integer[4]), 
##  'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
##  'dgs'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
##  'fSz'::(integer[4]), 
##  'fbuff'::(ARRAY(1 .. 'fSz', 'integer[4]')), 
##  'rootsPtr'::(ARRAY(1 .. 'N'+1, 'integer[4]')),
#######################################################     
DftEvalM_ALGEB := trampoline('DftMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn");

#######################################################     
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'es'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ESz'::(integer[4]), 
##  'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
##  'rootsPtr'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
#######################################################     
DftInterpM_ALGEB :=
    trampoline('InvDftMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"); 

#######################################################     
##  'N'::(integer[4]), 
##  'w'::(integer[4]), 
##  'SSz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 'SSz', 'integer[4]')), 
##  'Edims1'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
##  'E1Sz'::(integer[4]), 
##  'E1'::(ARRAY(1 .. 'E1Sz', 'integer[4]')), 
##  'Edims2'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
##  'E2Sz'::(integer[4]), 
##  'E2'::(ARRAY(1 .. 'E2Sz', 'integer[4]')), 
##  'p'::(integer[4]),  
##  'RETURN'::(integer[4]), 
#######################################################     
GetResChains_ALGEB :=
    trampoline('SubResultantChains_ALGEB', 'MAPLE', 'LIB' = "modpn"); 

#######################################################     
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'ith'::(integer[4]), 
##  'dth'::(integer[4]), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
##  'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
#######################################################     
DftInterpithdthM_ALGEB :=
    trampoline('InvDftIthDthMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"); 

#######################################################     
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'ith'::(integer[4]), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
##  'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
#######################################################     
DftInterpithM_ALGEB :=
    trampoline('InvDftIthMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"): 

#######################################################     
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'start'::(integer[4]), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
##  'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'RETURN'::(integer[4]),
#######################################################     
DftInterpNextDefectiveLCM_ALGEB := trampoline(
    'InvDftNextDefectiveLCMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"):

#######################################################     
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'start'::(integer[4]), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicees'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
##  'roots'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'RETURN'::(integer[4]),
#######################################################     
DftInterpNextLCM_ALGEB :=
    trampoline('InvDftNextLCMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"):


################################################################################
################################################################################
################################################################################
################################################################################
## subproduct tree based functions for constructing SCUBE, ALGEB version
################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
##  'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ESz'::(integer[4]), 
##  'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
##  'dgs'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
##  'fSz'::(integer[4]), 
##  'fbuff'::(ARRAY(1 .. 'fSz', 'integer[4]')), 
##  'RETURN'::(integer[4]), 
################################################################################
FastEvalM_ALGEB :=
    trampoline('FastEvalMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"); 

################################################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
##  'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ESz'::(integer[4]), 
##  'E'::(ARRAY(1 .. 'ESz', 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')),  
##  'RETURN'::(integer[4]), 
################################################################################
FastInterpM_ALGEB :=
    trampoline('FastInterpMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"); 

################################################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
##  'ith'::(integer[4]), 
##  'dth'::(integer[4]), 
##  'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')), 
################################################################################
FastInterpithdthM_ALGEB :=
    trampoline('InterpIthDthMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"); 

################################################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
##  'ith'::(integer[4]), 
##  'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 1, 'integer[4]')), 
################################################################################
FastInterpithM_ALGEB :=
    trampoline('InterpIthMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"): 

################################################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
##  'start'::(integer[4]), 
##  'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')),  
##  'RETURN'::(integer[4]),
################################################################################
FastInterpNextLCM_ALGEB :=
    trampoline('InterpNextLCMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"): 

################################################################################
##  'Nmp'::(ARRAY(1 .. 3, 'integer[4]')), 
##  'PHWD'::(ARRAY(1 .. 4, 'integer[4]')), 
##  'start'::(integer[4]), 
##  'bounds'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'ptss'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'hs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ws'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'NNs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Bs'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'datas'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'w'::(integer[4]), 
##  'slicesz'::(integer[4]), 
##  'slicedims'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'Ssz'::(integer[4]), 
##  'S'::(ARRAY(1 .. 1, 'integer[4]')), 
##  'dVsz'::(integer[4]), 
##  'pdegVec'::(ARRAY(1 .. 'dVsz', 'integer[4]')), 
##  'cVsz'::(integer[4]), 
##  'coefVec'::(ARRAY(1 .. 'cVsz', 'integer[4]')),  
##  'RETURN'::(integer[4]),
################################################################################
FastInterpNextDefectiveLCM_ALGEB :=
    trampoline('InterpNextDefectiveLCMultiWrapCN_ALGEB', 'MAPLE', 'LIB' = "modpn"): 

end module;
