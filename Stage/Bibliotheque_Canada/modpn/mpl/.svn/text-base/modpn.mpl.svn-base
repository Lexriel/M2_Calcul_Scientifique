#Error messages and message bound for Modpn library
macro(
    MEMORY_BOUND = 10000000000,
    ERR_FOURIERPRIME = "Not a Fourier Prime preferred by MODPN lib!",
    ERR_INTERRUPT="Program has been interrupted.",
    ERR_PRIME="Prime number is too small to handle the FFT-based computation, please choose a more suitable Fourier prime number.",
    ERR_MEMORY="Input problem is too big, software memory limit exceeded!"
):

modpn := module()
    #============================#
    #    EXPORTED FUNCTIONS      #
    #============================#
    ## submodules
    export ConnectorModule, ConnectorModuleNew1, RecdenConnector, MapleDagModule;

    ## polynomial, ring and data conversion
    export PolynomialRing, CRepAssert, CRepAssertUNI, PartialDeg,
    PartialDegsAssert, PolynomialConvertIn, PolynomialConvertOut,
    PolynomialConvertInUNI, PolynomialConvertOutUNI, Rpoly2Modpn,
    removeLeadingZeros, LeadingCoefUNI, EnlargePoly, calSizDgs,
    varno, getDGS;

    ## subproduct tree based functions for building subresultant chains
    export SubProductTreeCreate, CreatePointsTrees, GetResultantChains,
    FastInterpolationM, FastEvaluation, FastInterpolation, FastEvaluationM1,
    FastEvaluationM2, FastInterpolatePolynomial, FastInterpolateCoefficient,
    FastInterpolateNextLC, FastInterpolateNextDefectiveLC;

    ## subproduct tree based functions, ALGEB version
    export GetResultantChains_ALGEB, FastEvaluationM1_ALGEB,
    FastInterpolationM_ALGEB, FastEvaluationM2_ALGEB,
    FastInterpolatePolynomial_ALGEB, FastInterpolateCoefficient_ALGEB,
    FastInterpolateNextLC_ALGEB, FastInterpolateNextDefectiveLC_ALGEB;

    ## FFT based functions for building subresultant chains
    export GetResultantChainsDft, DftPreComp, DftEvaluationM,
    DftInterpolationM, DftInterpolatePolynomial, DftInterpolateCoefficient,
    DftInterpolateNextLC, DftInterpolateNextLT, DftInterpolateNextDefectiveLC;

    ## FFT based functions for building subresultant chains, ALGEB version
    export DftEvaluationM_ALGEB, DftInterpolationM_ALGEB,
    GetResultantChainsDft_ALGEB, DftInterpolatePolynomial_ALGEB,
    DftInterpolateCoefficient_ALGEB, DftInterpolateNextDefectiveLC_ALGEB,
    DftInterpolateNextLC_ALGEB; 

    ## TFT based functions for building subresultant chains
    export GetResultantChainsTFT, TFTPreComp, TFTEvaluationM, TFTInterpolationM, 
    TFTInterpolatePolynomial, TFTInterpolateCoefficient,
    TFTInterpolateNextDefectiveLC, TFTInterpolateNextLC;

    ## TFT based functions for building subresultant chains, ALGEB version
    export TFTEvaluationM_ALGEB, TFTInterpolationM_ALGEB,
    GetResultantChainsTFT_ALGEB, TFTInterpolatePolynomial_ALGEB,
    TFTInterpolateCoefficient_ALGEB, TFTInterpolateNextDefectiveLC_ALGEB,
    TFTInterpolateNextLC_ALGEB;

    ## Polynomial division
    export GetQuotientImage, DftGetQuotientImage, CreatePointsTreesDiv,
    FastInterpolatePolynomialDiv, DftPreCompDiv, FastInterpolatePolynomialDivDft,
    PlainPrem_rc, FastPrem_rc, PlainPrem_rc_rec, FastPrem_rc_rec,
    FastMonicDivUNI_rc;

    ## regularization and normalization
    export FastMonicDivUNI, PlainMonicDivUNI, FastPseudoRemainder,
    PlainPseudoRemainder, ReduceCoeffs, QuoModTriSet, IsInvertibleZeroDim,
    NormalForm, Normalize, MonicizeUNI, IsInvertable;

    ## gcd and resultant
    export PlainXGCDUNI, FastXGCDUNI, IteratedResultantOneDim,
    IteratedResultantZeroDim, RegularGcdZeroDim, MultivariateResultant;

    ## misc functions
    export MallocGcArray, FreeGcArray, version, TFTFFTMul, TFTFFTMulUNI,
    HenselLifting, FastInterpRatFunCons;

    ## CUDA functions
    export IsCUDAEnabled, EnableCUDA;

    ## Specialized Solver
    export BivariateSolve;
    
## PolynomialConvertIn PolynomialRing Rpoly2Modpn
## PolynomialConvertInUNI PolynomialConvertOut
## PolynomialConvertOutUNI TFTFFTMul TFTFFTMulUNI
## createZeroRecUni FastMonicDivUNI PlainMonicDivUNI
## LeadingCoefUNI DegreeUNI CoMulRPolyUNI MonicizeUNI
## PlainPseudoRemainder PlainXGCDUNI FastXGCDUNI PlainPrem_rc
## FastMonicDivUNI_rc
##
## SubProductTreeCreate FastEvaluation FastInterpolation
## FastInterpolationM DftInterpolationM FastEvaluationM1
## { The above 6 functions need type checking, wrappers }
## { Multivariate monicdivision, see FastTriade }
## { IsInvertable IsInvertibleZeroDim NormalForm Normalize, see FastTriade }
## { IteratedResultantZeroDim ReduceCoeffs QuoModTriSet, see FastTriade }

    #============================#
    #       LOCAL FUNCTIONS      #
    #============================#
    local shiftArrR, shiftArrL, addArr, calSizDgsfrom2,
    estimatePartialDegVecSize, estimatePartialDegVecSizefrom2, copyCRec,
    copyRec, createZeroRecUni, DegreeUNI, CoMulRPolyUNI, PlainXGCDUNIlocal,
    FastXGCDUNIlocal, log2Ceil, div2Ceil, estimateForOnePoly_Lift,
    getTriSetDegs, estimateThe2VecSiz_Lift, convertAndMergeSLG_Lift, 
    CRepAssertTriSet_Lift, MergeTriSetRep_Lift, MapTriSetX_Y2ONE_Lift_IN,
    MapTriSetX_Y2ONE_Lift_OUT, splitTriSetRep_Lift, CRepAssertTriSet, 
    MergeTriSetRep, getIndexOfVar, EnlargeRecden2N, splitTriSetRep,
    getAListOfVectors, Vec2Rep, estimateForOnePoly, estimateThe2VecSiz_TriSet, 
    MinimalDegreeBound;

    #=================================#
    #    modpn is a Maple package     #
    #=================================#
    option package;

#read in submodules
#uncomment savelib at end of file -> savelib('modpn') to save
#$include <ConnectorModule.mm>
#$include <ConnectorModuleNew1.mm>
#$include <RecdenConnector.mm>
#$include <MapleDagModule.mm>

# for Maplesoft tree
# also comment out savelib at end of file -> #savelib('modpn');
$include <modpn/src/ConnectorModule.mm>
$include <modpn/src/ConnectorModuleNew1.mm>
$include <modpn/src/RecdenConnector.mm>
$include <modpn/src/MapleDagModule.mm>

version := proc()
    return "2011.02.01";
end proc:

#------------------------------------------------------------------------------
# Function:
# { IsCUDAEnabled / EnableCUDA }
# Briefly:
# { enable/disable CUDA support}
# Calling sequence:
# { IsCUDAEnabled(); }
# { EnableCUDA(truefalse); }
# Input:
# { truefalse: boolean }
# Output:
# { IsCUDAEnabled() returns if CUDA is enabled for computations. }
# { EnableCUDA(true) to enable CUDA and return the previous status. }
# { EnableCUDA(false) to enable CUDA and returns the previous status. }
#------------------------------------------------------------------------------
IsCUDAEnabled := proc() :: truefalse;
local tag, major, minor, prop;
    prop := CUDA:-Properties(id = 0);
    major, minor := prop["Major"], prop["Minor"];
    if (major < 2 and minor < 2) then
        error "requires CUDA hardware with compute capability at least 1.2";
    else
        ## 0 stands for disabled
        ## 1 stands for enabled or (nonzero)
        tag := ConnectorModuleNew1:-IsCudaEnabledCN();
        return evalb(tag <> 0);
    end if;
end proc:

EnableCUDA := proc(flag::truefalse) :: truefalse;
local oldflag, tag;
    oldflag := IsCUDAEnabled();
    if (flag = oldflag) then return flag; end if;
    if (flag) then tag := 1 else tag := 0; end if;
    ConnectorModuleNew1:-EnableCudaCN(tag);
    return oldflag;
end proc:

#------------------------------------------------------------------------------
# Function:
# { BivariateSolve }
# Briefly:
# { Solve a 2 by 2 bivariate system over a finite field }
# Calling sequence:
# { BivariateSolve(F, G, R); }
# Input:
# { F : Polynomial }
# { G : Polynomial }
# { R : Polynomial ring }
# Output:
# { A list of pairs }
#------------------------------------------------------------------------------
BivariateSolve := proc(F, G, R)
local frep, grep, dec, v1, v2, L, A, B, i, result, n;
    
    frep := PolynomialConvertIn(R, F);
    grep := PolynomialConvertIn(R, G);
    frep := PartialDegsAssert(frep): 
    grep := PartialDegsAssert(grep): 
    frep := CRepAssert(frep, 1):
    grep := CRepAssert(grep, 1):
    ##printf("frep = %a\n", frep);
    ##printf("F = %a\n", Expand(F) mod R:-prime);
    ##printf("frep:-DGS = %a\n", frep:-DGS);
    ##printf("grep:-DGS = %a\n", grep:-DGS);
    ##printf("frep:-CRep:-COEF = %a\n", frep:-CRep:-COEF);
    ##printf("grep:-CRep:-COEF = %a\n", grep:-CRep:-COEF);

    ## Solving the system in C
    dec := ConnectorModuleNew1:-BivariateSolveCN_ALGEB("solve", frep:-DGS, 
        frep:-CRep:-COEF, grep:-DGS, grep:-CRep:-COEF, R:-prime);
    userinfo(3, SOLVE2, "solving in C", dec);
    
    ## Print the decomposition in C
    ##ConnectorModuleNew1:-BivariateSolveCN_ALGEB("print", dec);

    ############################################################################
    ## new conversion routine
    ############################################################################
    v1, v2 := R:-VarList[2], R:-VarList[1];
    ## the DAG ID is exprseq
    L := ConnectorModuleNew1:-BivariateSolveCN_ALGEB("build_dag", dec, [v1, v2]);
    L := [L];
    ASSERT(type(nops(L), even));

    ## build the result
    result := [];
    n := nops(L)/2;
    for i from 1 to n do
        A := L[2*i-1];
        B := L[2*i];
        if A = 0 then 
            result := [op(result), [B]];
        elif B = 0 then
            result := [op(result), [A]];
        else 
            result := [op(result), [A, B]];
        end if;
    end do:
    return result;

    ############################################################################
    ## old conversion routine, slow for large output decomposition
    ############################################################################
    #### Get the number of components
    ##n := ConnectorModuleNew1:-BivariateSolveCN_ALGEB("num_of_components", dec);
    ##userinfo(3, SOLVE2, "number of components is", n);
    ##if (n = 0) then return [] end if;
    ##
    #### Get the partial degree vector of the decomposition
    #### [ deg(A1, x), deg(B1, x), deg(B1, y),
    ####   ...
    ####   deg(An, x), deg(Bn, x), deg(Bn, y)]
    ##pdegsVec := Array(1..3*n, 'datatype' = 'integer[4]'):
    ##ConnectorModuleNew1:-BivariateSolveCN_ALGEB("deg_info", dec, pdegsVec);
    ##
    #### Convert the decomposition into a list of pairs as maple objects
    ##szsVec := Array(1..2*n, 'datatype' = 'integer[4]');
    ##for i from 1 to n do 
    ##    szsVec[2 * i - 1] := 1 + pdegsVec[3 * i - 2];
    ##    szsVec[2 * i] := (1 + pdegsVec[3 * i - 1]) * (1 + pdegsVec[3 * i]);
    ##end do:
    ##sz := add(szsVec[i], i = 1..2*n);
    ##coefsVec := Array(1..sz, 'datatype' = 'integer[4]'):
    ##ConnectorModuleNew1:-BivariateSolveCN_ALGEB("convert", dec, coefsVec);
    ##
    ##result := [];
    ##offset := 0;
    #### the variable ordering is VarList[1] > VarList[2]
    ##v1, v2 := R:-VarList[2], R:-VarList[1];
    ##for i from 1 to n do 
    ##    dA1, dB1, dB2 := pdegsVec[3*i-2], pdegsVec[3*i-1], pdegsVec[3*i];
    ##    ## construct polynomial A in the triangular set
    ##    A := 0; 
    ##    if (dA1 <> -1) then
    ##        coefA := Vector(coefsVec[offset+1..offset+szsVec[2*i-1]]);
    ##        A := PolynomialTools:-FromCoefficientVector(coefA, v1);
    ##    end if;
    ##
    ##    ## construct polynomial B in the triangular set
    ##    B := 0;
    ##    if (dB2 <> -1) then
    ##        offset := offset + szsVec[2*i-1];
    ##        coefB := Vector(coefsVec[offset+1..offset+szsVec[2*i]]);
    ##        offset := offset + szsVec[2*i];
    ##        L := [seq(PolynomialTools:-FromCoefficientVector
    ##                (coefB[(1+dB1)*i+1..(1+dB1)*i+1+dB1], v1), i=0..dB2)];
    ##        B := PolynomialTools:-FromCoefficientList(L, v2);
    ##    end if;
    ##
    ##    if A = 0 then 
    ##        result := [op(result), [B]];
    ##    elif B = 0 then
    ##        result := [op(result), [A]];
    ##    else 
    ##        result := [op(result), [A, B]];
    ##    end if;
    ##end do:
    ##return result;
end proc:
#------------------------------------------------------------------------------
# Function:
# { MallocGcArray/FreeGcArray}
# Briefly:
# { Allocate/delocate a C-array visible to gc }
# Calling sequence:
# { a := MallocGcArray(n); }
# { FreeGcArray(a); }
# Input:
# { n : positive integer }
# { a : MaplePointer }
# Output:
# { MallocGcArray(n) returns a MaplePointer to a sfixn C-array a 
#   of size n. If a = 0, then allocation failed. }
# { FreeGcArray(a) manually frees the memory allocated. }
# { If FreeGcArray(a) is not called, gc will free the memory in the end. }
#------------------------------------------------------------------------------
MallocGcArray := proc(n::posint)
local a;
   a := ConnectorModuleNew1:-MapleGcArray("malloc", n); 
   userinfo(4, MODPNGC, "a = ", a);
   return a;
end proc:

FreeGcArray := proc(a)
    if (a <> 0) then 
        return ConnectorModuleNew1:-MapleGcArray("free", a);
    else
        userinfo(4, MODPNGC, "try to free a null Gc-Array");    
    end if;
end proc:
#------------------------------------------------------------------------------
# Function: { TFTEvaluationM_ALGEB } 
# Briefly: { Evalute a polynomial at an array of points }
# Calling sequence:
# { TFTEvaluationM_ALGEB(N, M, in_rep, es, dims, ROOTS, p) }
# Input: 
# { N : index of the main variable }
# { M : number the variables that we specialize }
# { rep : Modpn polynomial }    
# { TFTPreRes : result from TFTPreComp }
# { p: Fourier prime }       
# Output: 
# { ECube, inside there is a MaplePointer pointers the evaluations }
#------------------------------------------------------------------------------
TFTEvaluationM_ALGEB := proc(N, M, ININrep, es, dims, ROOTS, p)
    local Ecube, Edat, Esz, NMp, i, pSz, rep, INrep;
    
    ASSERT(M<=N, "M is at most N", M, N);

    INrep := PartialDegsAssert(ININrep):
    rep := CRepAssert(INrep, 1):
    pSz := calSizDgs(rep:-DGS, N):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N:
    NMp[2] := M:
    NMp[3] := p:

    ## dims[1] is 0, and not in use
    Esz := mul(dims[i], i=2..N+1);
    if (Esz > MEMORY_BOUND) then error ERR_MEMORY end if;
    Edat := MallocGcArray(Esz);
    ## Edat should be a MaplePointer
    ASSERT(Edat <> 0, "fail to malloc an array in C");
    userinfo(8, MODPN, "MaplePointer created", Edat);
    
    ConnectorModuleNew1:-TFTEvalM_ALGEB(NMp, es, dims, Esz, Edat, rep:-DGS,
        pSz, rep:-CRep:-COEF, ROOTS);

    Ecube := Record('Ees', 'Edims', 'Esize', 'Edata'):
    Ecube:-Ees := es;
    Ecube:-Edims := dims;
    Ecube:-Esize := Esz;
    Ecube:-Edata := Edat;
    return Ecube;
end proc:

#------------------------------------------------------------------------------
# Function: { TFTInterpolationM_ALGEB }
# Briefly: { Interpolate a polynomial from an ECube. }
# Calling sequence:
# { TFTInterpolationM_ALGEB(R, N, M, ESTIDGS, Ecube, ROOTS, p); }
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables appearing }
# { M : the number of variables evaluated, should be N-1 }
# { R : modpn polynomial ring of characteristic p and with variable list }
# { ESTIDGS : esimate partial degrees for the output polynomial }
# { Edims, Esz, E : output of the TFT evaluation }
# Output:
# { a modpn multivariate polynomial in recden format }
#------------------------------------------------------------------------------
TFTInterpolationM_ALGEB := proc(R, N, M, ESTIDGS, Ecube, ROOTS, p)
local NMp, cVsz, coefVec, dVsz, pdegVec, rep;
    dVsz := estimatePartialDegVecSize(ESTIDGS, N) + 1;
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY end if;
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]');
    cVsz := calSizDgs(ESTIDGS, N) + 1;
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY end if;
    coefVec := Array(1..cVsz, 'datatype'='integer[4]');
    NMp := Array(1..3, 'datatype' = 'integer[4]');
    NMp[1] := N; 
    NMp[2] := M; 
    NMp[3] := p;
    ASSERT(Ecube:-Edata <> 0, "MaplePointer Edata should not be NULL");
    ConnectorModuleNew1:-TFTInterpM_ALGEB(NMp, Ecube:-Ees, Ecube:-Edims,
        Ecube:-Esize, Ecube:-Edata, dVsz, pdegVec, cVsz, coefVec, ROOTS);

    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase');
    rep:-RecdenPoly :=
        RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList);
    rep:-DGS := 0;
    rep:-CRep := Record('COEF', 'PDGSV', 'COEFV');
    rep:-CRep:-COEF := 0;
    rep:-CRep:-PDGSV := pdegVec;
    rep:-CRep:-COEFV := coefVec;
    rep:-CRepCase := 2;
    return rep;
end proc:
#------------------------------------------------------------------------------
# Function: { GetResultantChainsTFT_ALGEB }
# Briefly:  { Build the SCube of two polynomials }
# Calling sequence:
# { GetResultantChainsTFT_ALGEB(N, M, w, bounds, Ecube1, Ecube2, p);
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables}
# { M : the number of variables to be evaluated, should be N-1 }
# { w : the minimum degree of input polynomials wrt the (common) main variable. }
# { bounds : a one-dimensional array containing (plus-one) bounds for
#   the resultant of the input polynomials w.r.t. their other
#   variables, that is, X_1, ..., X_M. Note the first slot is not used. }
# { Ecube1: the evaluation representation of the first polynomial. }
# { Ecube2: the evaluation representation of the second polynomial. }
# Assumptions:
# { Edata in Ecube1 and Ecube2 are MaplePointers. }
# Output:
# { Scube : the subresultant chains of the input polynomials evaluated
#           at each point of their evaluation representation. }
# { Sdims : the dimensions of S, that is, 'bounds' followed by '[w, w]'. }
# { Ssz   : the total number of slots in 'Sdat'. }
# { Sdat  : MaplePointer object. } 
#------------------------------------------------------------------------------
GetResultantChainsTFT_ALGEB := proc(N, M, w, bounds, Ecube1, Ecube2, p)
local mymsg, Scube, Sdat, Ssz, i, ssdims;

    Ssz := mul(bounds[i], i=2..M+1);
    Ssz := Ssz*w*w:
    if (Ssz > MEMORY_BOUND) then error ERR_MEMORY: end if:

    Sdat := MallocGcArray(Ssz);
    ASSERT(Sdat<>0, "fail the allocate C array of size", Ssz);

    userinfo(8, MODPN, "Sdat=", Sdat);

    mymsg := ConnectorModuleNew1:-GetResChainsTFT_ALGEB(N, M, w, Ssz, Sdat, 
                      Ecube1:-Edims, Ecube1:-Esize, Ecube1:-Edata, 
                      Ecube2:-Edims, Ecube2:-Esize, Ecube2:-Edata, p):

    if (mymsg=-10)  then error ERR_INTERRUPT: end if:
    if (mymsg=-100) or (mymsg=-110) then error ERR_PRIME: end if:
    if (mymsg=-200) then error ERR_FOURIERPRIME: end if:
    ssdims := Array(1..N+2, 'datatype'='integer[4]'):
    for i from 2 to N do
        ssdims[i] := bounds[i];
    end do;

    ssdims[N+1] := w;
    ssdims[N+2] := w;
    Scube := Record('Ses', 'Sdims', 'Ssize', 'Sdata', 'SLc', 'SPoly');
    Scube:-Ses := Ecube1:-Ees;
    Scube:-Sdims := ssdims;
    Scube:-Ssize := Ssz;
    Scube:-Sdata := Sdat;
    Scube:-SLc := Array(1..w);
    Scube:-SPoly := Array(1..w);
    return Scube;
end proc:
#------------------------------------------------------------------------------
# Function: { TFTInterpolatePolynomial_ALGEB }
# Briefly: { Interpolate ith subresultant from a SCUBE }
# Calling sequence:
# { TFTInterpolatePolynomial_ALGEB(R, N, M, ith, w, Scube, ROOTS, p) }
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { ith: ith row in the subresultant chain. }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChainsTFT_ALGEB }
# { ROOTS : primitive roots used for evaluations }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the i-th subresultant }
#------------------------------------------------------------------------------
TFTInterpolatePolynomial_ALGEB := proc(R, N, M, ith, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep,
    slicedims, slicees, slicesz;

    if(((Scube:-SPoly)[ith+1]) <> 0) then return (Scube:-SPoly)[ith+1] end if;

    slicees := Scube:-Ses;
    slicedims := Scube:-Sdims;
    Ssz := Scube:-Ssize;
    S := Scube:-Sdata;
    ASSERT(S<>0, "NULL MaplePointer not expected", S);
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]');
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] - 1 end do;
    ESTIDGS[N+1] := w-1;
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1);
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]');
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1;
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY end if;
    coefVec := Array(1..cVsz, 'datatype'='integer[4]');
    NMp := Array(1..3, 'datatype' = 'integer[4]');
    NMp[1] := N;
    NMp[2] := M;
    NMp[3] := p;
    slicesz := iquo(Ssz, w);
    ConnectorModuleNew1:-TFTInterpithM_ALGEB(NMp, ith, w, slicesz, slicees,
        slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS);
    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase');
    rep:-RecdenPoly :=
        RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList);
    rep:-DGS := 0;
    rep:-CRep := 0;
    rep:-CRepCase := -1;

    if(nargs=9) then
        rep := ReduceCoeffs(R, rep, args[9]);
    end if:
    (Scube:-SPoly)[ith+1] := rep;
    return rep;
end proc:
#------------------------------------------------------------------------------
# Function: { TFTInterpolateCoefficient_ALGEB }
# Briefly: { Interpolate coefficient in the ith row and dth column of a SCUBE }
# Calling sequence:
# { TFTInterpolateCoefficient_ALGEB(R, N, M, ith, dth, w, Scube, ROOTS, p); }
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { ith: ith row in the subresultant chain. }
# { dth: dth column in the subresultant chain. }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChainsTFT_ALGEB }
# { ROOTS : primitive roots used for evaluations }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the coefficient in the i-th row and d-th column. }
#------------------------------------------------------------------------------
TFTInterpolateCoefficient_ALGEB := proc(R, N, M, ith, dth, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep, slicedims,
    slicees, slicesz, tmp, ww;

    slicees := Scube:-Ses;
    slicedims := Scube:-Sdims;
    Ssz := Scube:-Ssize;
    S := Scube:-Sdata;
    ASSERT(S<>0, "NULL MaplePointer not expected", S);
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]');
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] - 1 end do;
    ESTIDGS[N+1] := 0;
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1) + 1;
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY end if;
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]');
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1;
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY end if;
    coefVec := Array(1..cVsz, 'datatype'='integer[4]');
    NMp := Array(1..3, 'datatype' = 'integer[4]');
    NMp[1] := N;
    NMp[2] := M; 
    NMp[3] := p;
    ww := w * w;
    slicesz := iquo(Ssz, ww);

    tmp := slicedims[N+1];
    slicedims[N+1] := 1;
    ConnectorModuleNew1:-TFTInterpithdthM_ALGEB(NMp, ith, dth, w, slicesz,
        slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS);
    slicedims[N+1] := tmp;

    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):

    rep:-RecdenPoly :=
        RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):

    rep:-DGS := 0;
    rep:-CRep := Record('COEF', 'PDGSV', 'COEFV');
    rep:-CRep:-COEF := 0;
    rep:-CRep:-PDGSV := pdegVec;
    rep:-CRep:-COEFV := coefVec;
    rep:-CRepCase :=2;
    return rep;
end proc:
#------------------------------------------------------------------------------
# Function: { TFTInterpolateNextDefectiveLC_ALGEB }
# Briefly: { Interpolate the next defective leading coefficient in a SCUBE }
# Calling sequence: { }
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { start: index of a subresultant }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChainsTFT_ALGEB }
# { ROOTS : primitive roots used for evaluations }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the next defective leading coefficient from
#   the subresultant start }
#------------------------------------------------------------------------------
TFTInterpolateNextDefectiveLC_ALGEB := proc(R, N, M, start, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, j, pdegVec, rep, rep2,
    slicedims, slicees, slicesz, theend, tmp, ww, thestart;

    if start = w-1 then
        rep2 := Record('ith', 'LC');
        rep2:-ith := -1;
        rep2:-LC := 0;
        return rep2;
    end if;

    if (start<0) then thestart := 0; else thestart := start; end if:

    slicees := Scube:-Ses;
    slicedims := Scube:-Sdims;
    Ssz := Scube:-Ssize;
    S := Scube:-Sdata;
    ASSERT(S<>0, "NULL MaplePointer not expected", S);
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]');
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] - 1 end do;
    ESTIDGS[N+1] := 0;
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1) + 1;
    if(dVsz > MEMORY_BOUND) then error ERR_MEMORY end if;
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]');
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1;
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY end if;
    coefVec := Array(1..cVsz, 'datatype'='integer[4]');
    NMp := Array(1..3, 'datatype' = 'integer[4]');
    NMp[1] := N; 
    NMp[2] := M; 
    NMp[3] := p;
    ww := w * w;
    slicesz := iquo(Ssz, ww);
    tmp := slicedims[N+1];
    slicedims[N+1] := 1;
    theend := ConnectorModuleNew1:-TFTInterpNextDefectiveLCM_ALGEB(NMp, start, w,
        slicesz, slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS);
    slicedims[N+1] := tmp;

    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase');
    if theend = -1 then
        rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase');
        rep := 0;
        rep2 := Record('ith', 'LC');
        rep2:-ith := theend;
        rep2:-LC := rep;
    else
        rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase');
        rep:-RecdenPoly :=
            RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
        rep:-DGS := 0;
        rep:-CRep := Record('COEF', 'PDGSV', 'COEFV');
        rep:-CRep:-COEF := 0;
        rep:-CRep:-PDGSV := pdegVec;
        rep:-CRep:-COEFV := coefVec;
        rep:-CRepCase := 2;
        rep2 := Record('ith', 'LC');
        rep2:-ith := theend;
        rep2:-LC := rep;
    end if;
    return rep2;
end proc:

#------------------------------------------------------------------------------
# Function: { TFTInterpolateNextLC_ALGEB }
# Briefly: { Interpolate the next leading coefficient in a SCUBE }
# Calling sequence: { }
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { start: index of a subresultant }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChainsTFT_ALGEB }
# { ROOTS : primitive roots used for evaluations }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the next leading coefficient from the subresultant start }
#------------------------------------------------------------------------------
TFTInterpolateNextLC_ALGEB := proc(R, N, M, start, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, j, pdegVec, rep, rep2,
    slicedims, slicees, slicesz, theend, tmp, ww, thestart;

    if start = w-1 then
       rep2 := Record('ith', 'LC');
       rep2:-ith := -1;
       rep2:-LC := 0;
       return rep2;
    end if:

    if (start<0) then thestart := 0; else thestart := start; end if;
    for j from thestart+1 to w-1 do
        if(((Scube:-SLc)[j+1])<>0) then return (Scube:-SLc)[j+1]; end if;
    end do;

    slicees := Scube:-Ses;
    slicedims := Scube:-Sdims;
    Ssz := Scube:-Ssize;
    S := Scube:-Sdata;
    ASSERT(S<>0, "NULL MaplePointer not expected", S);
    ESTIDGS := Array(1..N+1, 'datatype'='integer[4]');
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] - 1 end do;
    ESTIDGS[N+1] := 0;
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1) + 1;
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY end if;
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]');
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1;
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY; end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]');
    NMp := Array(1..3, 'datatype' = 'integer[4]');
    NMp[1] := N; 
    NMp[2] := M; 
    NMp[3] := p;
    ww := w * w;
    slicesz := iquo(Ssz, ww);
    tmp := slicedims[N+1];
    slicedims[N+1] := 1;

    theend := ConnectorModuleNew1:-TFTInterpNextLCM_ALGEB(NMp, start, w, slicesz,
        slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS);
    slicedims[N+1] := tmp;

    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase');

    if theend = -1 then
        rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase');
        rep := 0;
        rep2 := Record('ith', 'LC');
        rep2:-ith := theend;
        rep2:-LC := rep;
    else
        rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase');
        rep:-RecdenPoly :=
            RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList);
        rep:-DGS := 0;
        rep:-CRep := Record('COEF', 'PDGSV', 'COEFV');
        rep:-CRep:-COEF := 0;
        rep:-CRep:-PDGSV := pdegVec;
        rep:-CRep:-COEFV := coefVec;
        rep:-CRepCase := 2;
        rep2 := Record('ith', 'LC');
        rep2:-ith := theend;
        rep2:-LC := rep;
        (Scube:-SLc)[(rep2:-ith) + 1] := rep2;
    end if;
    return rep2;
end proc:

################################################################################
################################################################################
################################################################################
##   FFT based functions for constructing SCUBE, ALGEB version   
################################################################################
################################################################################
################################################################################

#------------------------------------------------------------------------------
# Function: { DftEvaluationM_ALGEB }
# Briefly:
# { Evaluate a polynomial at primitive roots of unities }
# Input:
# { N: index of the main variable }
# { M: number the variables that we specialize (should N-1) }
# { ININrep: polynomial
# { es: array of Fourier degrees
# { dims: array of FFT sizes
# { ROOTS : array of primitive roots from least variable (index = 1) to top
#           evaluated variable (index = M) }
# Output:
# { ECube, evaluation cube, one of its fields is a MaplePointer. }
#------------------------------------------------------------------------------
DftEvaluationM_ALGEB := proc(N, M, ININrep, es, dims, ROOTS, p)
    local Ecube, Edat, Esz, NMp, i, pSz, rep, INrep;

    INrep := PartialDegsAssert(ININrep):
    rep := CRepAssert(INrep, 1):
    pSz := calSizDgs(rep:-DGS, N):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N: 
    NMp[2] := M: 
    NMp[3] := p:
    Esz := mul(dims[i], i=2..N+1);

    if (Esz > MEMORY_BOUND) then error ERR_MEMORY end if;
    Edat := MallocGcArray(Esz);
    ASSERT(Edat<>0, "fail to allocate Edat array in C");

    ConnectorModuleNew1:-DftEvalM_ALGEB(NMp, es, dims, Esz, Edat, rep:-DGS,
        pSz, rep:-CRep:-COEF, ROOTS):

    Ecube := Record('Ees', 'Edims', 'Esize', 'Edata'):
    Ecube:-Ees := es;
    Ecube:-Edims := dims;
    Ecube:-Esize := Esz;
    ## size of Edat:
    Ecube:-Edata := Edat:
    ## data array, its first position is 1.
    return Ecube:
end proc:

#------------------------------------------------------------------------------
# Function: { DftInterpolationM_ALGEB }
# Briefly: { Interpolate a polynomial from an ECube. }
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables appearing }
# { M : the number of variables to be evaluated }
# { R : a modpn polynomial ring of characteristic `p` and with variable list }
# { ESTIDGS : esimate partial degrees for the output polynomial }
# { Edims, Esz, E : output of the evaluation }
# Output:
# { a modpn multivariate polynomial in recden format }
#------------------------------------------------------------------------------
DftInterpolationM_ALGEB := proc(R, N, M, ESTIDGS, Ecube, ROOTS, p)
    local NMp, cVsz, coefVec, dVsz, pdegVec, rep;

    dVsz := estimatePartialDegVecSize(ESTIDGS, N)+1:
    if (dVsz>MEMORY_BOUND) then error ERR_MEMORY end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]');
    cVsz := calSizDgs(ESTIDGS, N) + 1:
    if (cVsz>MEMORY_BOUND) then error ERR_MEMORY end if;
    coefVec := Array(1..cVsz, 'datatype'='integer[4]');
    NMp := Array(1..3, 'datatype' = 'integer[4]');
    NMp[1] := N; 
    NMp[2] := M; 
    NMp[3] := p;
    ConnectorModuleNew1:-DftInterpM_ALGEB(NMp, Ecube:-Ees, Ecube:-Edims,
        Ecube:-Esize, Ecube:-Edata, dVsz, pdegVec, cVsz, coefVec, ROOTS);
    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase');
    rep:-RecdenPoly :=
        RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList);
    rep:-DGS := 0;
    rep:-CRep := Record('COEF', 'PDGSV', 'COEFV');
    rep:-CRep:-COEF := 0;
    rep:-CRep:-PDGSV := pdegVec;
    rep:-CRep:-COEFV := coefVec;
    rep:-CRepCase :=2;
    return rep;
end proc;

#------------------------------------------------------------------------------
# Function: { GetResultantChainsDft_ALGEB }
# Briefly:  { Build the SCube of two polynomials }
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables}
# { M : the number of variables to be evaluated, should be N-1 }
# { w : the minimum degree of input polynomials wrt the (common) main variable. }
# { bounds : a one-dimensional array containing (plus-one) bounds for
#   the resultant of the input polynomials w.r.t. their other
#   variables, that is, X_1, ..., X_M. Note the first slot is not used. }
# { Ecube1: the evaluation representation of the first polynomial. }
# { Ecube2: the evaluation representation of the second polynomial. }
# Assumptions:
# { Edata in Ecube1 and Ecube2 are MaplePointers. }
# Output:
# { Scube : the subresultant chains of the input polynomials evaluated
#           at each point of their evaluation representation. }
# { Sdims : the dimensions of S, that is, 'bounds' followed by '[w, w]'. }
# { Ssz   : the total number of slots in 'Sdat'. }
# { Sdat  : MaplePointer object. } 
#------------------------------------------------------------------------------
GetResultantChainsDft_ALGEB := proc(N, M, w, bounds, Ecube1, Ecube2, p)
    local mymsg, Scube, Sdat, Ssz, i, ssdims;

    Ssz := mul(bounds[i], i=2..M+1);
    Ssz := w*w*Ssz;

    if (Ssz > MEMORY_BOUND)  then error ERR_MEMORY: end if:

    Sdat := MallocGcArray(Ssz);
    ASSERT(Sdat<>0, "NULL MaplePointer not expected");
    
    mymsg := ConnectorModuleNew1:-GetResChains_ALGEB(N, w, Ssz, Sdat, 
        Ecube1:-Edims, Ecube1:-Esize, Ecube1:-Edata, Ecube2:-Edims,
        Ecube2:-Esize, Ecube2:-Edata, p):

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error ERR_PRIME: end if:
    if (mymsg = -200) then error ERR_FOURIERPRIME: end if:
    ssdims := Array(1..N+2, 'datatype'='integer[4]'):
    for i from 2 to N do ssdims[i] := bounds[i]: end do:
    ssdims[N+1] := w:
    ssdims[N+2] := w:
    Scube := Record('Ses', 'Sdims', 'Ssize', 'Sdata', 'SLc', 'SPoly'):
    Scube:-Ses := Ecube1:-Ees;
    Scube:-Sdims := ssdims:
    Scube:-Ssize := Ssz:
    Scube:-Sdata := Sdat:
    Scube:-SLc := Array(1..w):
    Scube:-SPoly := Array(1..w):
    return Scube:
end proc:

#------------------------------------------------------------------------------
# Function: { DftInterpolatePolynomial_ALGEB }
# Briefly: { Interpolate ith subresultant from a SCUBE, ALGEB version }
# Calling sequence:
# { DftInterpolatePolynomial_ALGEB(R, N, M, ith, w, Scube, ROOTS, p) }
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { ith: ith row in the subresultant chain. }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChainsTFT_ALGEB }
# { ROOTS : primitive roots used for evaluations }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the i-th subresultant }
#------------------------------------------------------------------------------
DftInterpolatePolynomial_ALGEB:=proc(R, N, M, ith, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep,
    slicedims, slicees, slicesz;

    if(((Scube:-SPoly)[ith+1]) <> 0) then
         return (Scube:-SPoly)[ith+1]:
    end if:

    slicees := Scube:-Ses:
    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ASSERT(S<>0, "NULL MaplePointer not expected");

    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] - 1 end do:
    ESTIDGS[N+1] := w-1:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N:
    NMp[2] := M:
    NMp[3] := p:
    slicesz := iquo(Ssz, w):

    ConnectorModuleNew1:-DftInterpithM_ALGEB(NMp, ith, w, slicesz, slicees,
        slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS);

    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly := RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
    rep:-DGS := 0:
    rep:-CRep := 0:
    rep:-CRepCase := -1:

    if (nargs=9) then rep := ReduceCoeffs(R, rep, args[9]): end if:
    (Scube:-SPoly)[ith+1] := rep:
    return rep:
end proc:

#------------------------------------------------------------------------------
# Function: { DftInterpolateCoefficient_ALGEB }
# Briefly: { Interpolate coefficient in the ith row and dth column of a SCUBE }
# Calling sequence:
# { DftInterpolateCoefficient_ALGEB(R, N, M, ith, dth, w, Scube, ROOTS, p); }
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { ith: ith row in the subresultant chain. }
# { dth: dth column in the subresultant chain. }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChainsTFT_ALGEB }
# { ROOTS : primitive roots used for evaluations }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the coefficient in the i-th row and d-th column. }
#------------------------------------------------------------------------------
DftInterpolateCoefficient_ALGEB := proc(R, N, M, ith, dth, w, Scube, ROOTS,  p)
    local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep,
    slicedims, slicees, slicesz, tmp, ww;

    slicees := Scube:-Ses:
    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ASSERT(S<>0, "NULL MaplePointer not expected");

    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] - 1 end do;
    ESTIDGS[N+1] := 0:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1) + 1:
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]');
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N;
    NMp[2] := M;
    NMp[3] := p;
    ww := w*w;
    slicesz := iquo(Ssz, ww):

    tmp := slicedims[N+1]:
    slicedims[N+1] := 1:
    ConnectorModuleNew1:-DftInterpithdthM_ALGEB(NMp, ith, dth, w, slicesz, slicees, 
        slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    slicedims[N+1]:=tmp:

    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly := RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
    rep:-DGS := 0:
    rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF := 0:
    rep:-CRep:-PDGSV := pdegVec:
    rep:-CRep:-COEFV := coefVec:
    rep:-CRepCase := 2:
    return rep:
end proc:

#------------------------------------------------------------------------------
# Function: { DftInterpolateNextDefectiveLC_ALGEB }
# Briefly: { Interpolate the next defective leading coefficient in a SCUBE }
# Calling sequence: { }
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { start: index of a subresultant }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChainsTFT_ALGEB }
# { ROOTS : primitive roots used for evaluations }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the next defective leading coefficient from
#   the subresultant start }
#------------------------------------------------------------------------------
DftInterpolateNextDefectiveLC_ALGEB := proc(R, N, M, start, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz,cVsz, coefVec, dVsz, i, j, pdegVec, rep, rep2,
    slicedims, slicees, slicesz, theend, tmp, ww, thestart;

    if start = w-1 then
       rep2 := Record('ith', 'LC');
       rep2:-ith := -1;
       rep2:-LC := 0:
       return rep2:
    end if:

    if (start<0) then thestart := 0: else thestart := start: end if:

    #for j from thestart+1 to w-1 do
    #   if(((Scube:-SLc)[j+1])<>0) then
    #      return (Scube:-SLc)[j+1]:
    #   end if:
    #end do:

    slicees := Scube:-Ses:
    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ASSERT(S<>0, "NULL MaplePointer not expected");
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] -1 end do;
    ESTIDGS[N+1] := 0:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1) + 1:
    if (dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N:
    NMp[2] := M:
    NMp[3] := p:
    ww := w*w;
    slicesz := iquo(Ssz, ww):
    tmp := slicedims[N+1]:
    slicedims[N+1] := 1:
    theend := ConnectorModuleNew1:-DftInterpNextDefectiveLCM_ALGEB(NMp, start, w,
        slicesz, slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    slicedims[N+1] := tmp:

    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):

    if theend = -1 then
      rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep := 0;
      rep2 := Record('ith', 'LC');
      rep2:-ith := theend;
      rep2:-LC := rep:
    else
      rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:-RecdenPoly := RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
      rep:-DGS := 0:
      rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
      rep:-CRep:-COEF := 0:
      rep:-CRep:-PDGSV := pdegVec:
      rep:-CRep:-COEFV := coefVec:
      rep:-CRepCase := 2:
      rep2 := Record('ith', 'LC');
      rep2:-ith := theend;
      rep2:-LC := rep:
    end if:
    return rep2:
end proc:

#------------------------------------------------------------------------------
# Function: { DftInterpolateNextLC_ALGEB }
# Briefly: { Interpolate the next leading coefficient in a SCUBE }
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { start: index of a subresultant }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChainsTFT_ALGEB }
# { ROOTS : primitive roots used for evaluations }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the next leading coefficient from the subresultant start }
#------------------------------------------------------------------------------
DftInterpolateNextLC_ALGEB := proc(R, N, M, start, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz,cVsz, coefVec, dVsz, i, j, pdegVec, rep,
    rep2, slicedims, slicees, slicesz, theend, tmp, ww, thestart;

    if start = w-1 then
       rep2 := Record('ith', 'LC');
       rep2:-ith := -1;
       rep2:-LC := 0:
       return rep2:
    end if:
    
    if (start<0) then thestart := 0: else thestart := start: end if:
    
    for j from thestart+1 to w-1 do
       if(((Scube:-SLc)[j+1])<>0) then return (Scube:-SLc)[j+1]: end if:
    end do:
    
    slicees := Scube:-Ses:
    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ASSERT(S<>0, "NULL MaplePointer not expected");

    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] -1 end do;
    ESTIDGS[N+1] := 0:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1) + 1:
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N:
    NMp[2] := M:
    NMp[3] := p:
    ww := w*w;
    slicesz := iquo(Ssz, ww):

    tmp := slicedims[N+1]:
    slicedims[N+1] := 1:
    theend := ConnectorModuleNew1:-DftInterpNextLCM_ALGEB(NMp, start, w, slicesz,
        slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    slicedims[N+1] := tmp:
    
    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    if theend = -1 then
      rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep := 0;
      rep2 := Record('ith', 'LC');
      rep2:-ith := theend;
      rep2:-LC := rep:
    else
      rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:-RecdenPoly := RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
      rep:-DGS := 0:
      rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
      rep:-CRep:-COEF := 0:
      rep:-CRep:-PDGSV := pdegVec:
      rep:-CRep:-COEFV := coefVec:
      rep:-CRepCase := 2:
      rep2 := Record('ith', 'LC');
      rep2:-ith := theend;
      rep2:-LC := rep:
      (Scube:-SLc)[(rep2:-ith) + 1] := rep2:
    end if:
    return rep2:
end proc:

################################################################################
################################################################################
################################################################################
##   TFT based functions for constructing SCUBE, non-ALGEB version   
################################################################################
################################################################################
################################################################################

#------------------------------------------------------------------------------
# Function: MinimalDegreeBound
# Briefly: Estimate the degree bound of the resultant
# Calling sequence:
# { MinimalDegreeBound(f, g, x, v) }
# Input : f, polynomial such that degree(f, x)>0
#         g, polynomial such that degree(g, x)>0
#         v, variable other than x
# Output : The degree bound of the resultant res(f, g, x) in variable v. 
#          That is, a number B such that degree(res(f, g, x), v) <= B  
#------------------------------------------------------------------------------
MinimalDegreeBound := proc(f, g, x, v) 
local td1, td2, d1, d2, Bbez, Brow, Bcol, i, j, k, dif, dig, PDF, PDG;
    if evalb(x=v) then
        userinfo(8, MODPN, "same variables occur", x, v);
        return 0; 
    end if;

    ## Bezout bound, better if f and g are dense
    td1 := degree(f); ## total degree, not patrial
    td2 := degree(g);    
    ASSERT(td1 > 0, "positive total degree expected", td1);
    ASSERT(td2 > 0, "positive total degree expected", td2);
    Bbez := td1 * td2;
    userinfo(8, MODPN, "bezout bound", v, Bbez);

    ## Sylvester Matrix based bound, better if f and g are sparse
    d1 := degree(f, x); ## partial degree, not total degree
    d2 := degree(g, x);
    ASSERT(d1 > 0, "positive partial degree expected", d1);
    ASSERT(d2 > 0, "positive partial degree expected", d2);

    ## Row bound
    Brow := degree(f, v) * d2 + degree(g, v) * d1;
    userinfo(8, MODPN, "   row bound", v, Brow);

    if (td1 = d1) and (td2 = d2) then
        ## userinfo(8, MODPN, "Should be a dense input"); 
        return min(Bbez, Brow);
    end if;

    ## Column bound computation

    ## t1 := time();   
    PDF := [seq(degree(coeff(f, x, j), v), j = 0..d1)];
    PDG := [seq(degree(coeff(g, x, j), v), j = 0..d2)];
    ## t2 := time();
    ##userinfo(9, MODPN, "Time to obtain partial degrees", t2 - t1);

    Bcol := 0;
    for i from 1 to d1 + d2 do
        dif := 0;
        ## upper-part, coefficients from f
        for k from 1 to d2 do
            j := d1 - i + k;
            if evalb(j<=d1) and evalb(j>=0) then 
                ## coeff(f, x, j) = coeff(f, x, d1 - i + k) is the ith column
                dif := max(PDF[j+1], dif); 
            end if;
        end do;
        ## lower-part, coefficients from g
        dig := 0;
        for k from 1 to d1 do
            j := d2 - i + k;
            if evalb(j<=d2) and evalb(j>=0) then 
                ## coeff(g, x, j) = coeff(g, x, d1 - i + k) is the ith column
                dig := max(PDG[j+1], dig); 
            end if;
        end do;
        userinfo(11, MODPN, "The", i, "-th column, upper part", dif);
        userinfo(11, MODPN, "The", i, "-th column, lower part", dig);
        userinfo(11, MODPN, "The", i, "-th column", max(dif, dig));
        Bcol := Bcol + max(dif, dig);
    end do;
    ## t3 := time();
    ##userinfo(9, MODPN, "Time to obtain column degree bound", t3 - t1);
    userinfo(8, MODPN, "column bound", v, Bcol);
    userinfo(200, MODPN, "True degree", v, degree(resultant(f, g, x), v));

    return min(Bbez, Brow, Bcol);
end proc:
#------------------------------------------------------------------------------
# Function: 
# Briefly: Prepare data for building a SCube
# Calling sequence:
# { TFTPreComp(N, M, ININrep1, ININrep2, p) }
# Input: 
# { N : index of the main variable }
# { M : number the variables that we specialize }
# { p1 : Modpn polynomial }    
# { p2 : Modpn polynomial }
# { p : Fourier prime }       
# Output : 
# { A record holding: 
#          Record( 'TIMES', 'ES', 'BOUNDS', 'DIMS1', 'DIMS2', 'ROOTS'):
#          ES     ## Fourier degrees that we need (in each dimension)
#          BOUNDS ## partial degree bound of the resultant
#          DIMS1  ## size in each dimension;
#                 ## DIMS1[2],...,DIMS1[M+1] are the TFT sizes;
#                 ## DIMS1[M+2],...,DIMS1[N+1] are the partial degree. 
#          DIMS2  ## size in each dimension, 
#          ROOTS  ## the prmitive roots (one in each dimension)
#                 ## The size is 2^ES[2] + 2^ES[3] + ... + 2^ES[M+1]
#          TIMES  ## Number of times to find good roots.
# } 
#------------------------------------------------------------------------------
TFTPreComp := proc(N, M, p1, p2, p)
    local mymsg, Res, bounds, dims1, dims2, es, i, m, ls,
          rep1, rep2, uroots, times, INrep1, INrep2, V1, V2;

    ASSERT(M<=N, "M should be at most N", M, N);

    INrep1 := PartialDegsAssert(p1);
    INrep2 := PartialDegsAssert(p2);
    rep1   := CRepAssert(INrep1, 1);
    rep2   := CRepAssert(INrep2, 1);
    es     := Array(1..M+1, 'datatype' = 'integer[4]');
    bounds := Array(1..M+1, 'datatype' = 'integer[4]');
    dims1  := Array(1..N+1, 'datatype' = 'integer[4]');
    dims2  := Array(1..N+1, 'datatype' = 'integer[4]');

    ## DGS[N] is the partial degree in the main variable
    for i from M+2 to N+1 do
        dims1[i] := rep1:-DGS[i-1] + 1;
        dims2[i] := rep2:-DGS[i-1] + 1;
    end do:

    ## Variable list
    ## rep1 and rep2 should have the same variable list.
    V1 := op(2, op(1, rep1[RecdenPoly]));
    V2 := op(2, op(1, rep2[RecdenPoly]));
    ASSERT(nops(V1) = nops(V2), "the same list of variables expected", V1, V2);
    for i from 1 to nops(V1) do
        ASSERT( V1[i] = V2[i], "the same list of variables expected", V1[i], V2[i]);
    end do;

    for i from 2 to M+1 do
        bounds[i] := MinimalDegreeBound(PolynomialConvertOut(rep1),
                     PolynomialConvertOut(rep2), V1[1], V1[-i+1]) + 1;
        userinfo(8, MODPN, "WAS", rep1:-DGS[i-1]*(rep2:-DGS[N])
                                  + rep2:-DGS[i-1]*(rep1:-DGS)[N] + 1);
        userinfo(8, MODPN, "NOW", bounds[i]);
        es[i] := log2Ceil(bounds[i]);
        dims1[i] := bounds[i];
        dims2[i] := 2^es[i]: ## Using dims2 as a temp, to assign later.
    end do:

    m := add(dims2[i], i=2..M+1);
    if (m > MEMORY_BOUND) then error ERR_MEMORY end if:
    uroots := Array(1..m, 'datatype' = 'integer[4]'):
    
    times := ConnectorModuleNew1:-createGoodRoots(N, M, rep1:-DGS, rep1:-CRep:-COEF,
        rep2:-DGS, rep2:-CRep:-COEF, es, dims2, uroots, p):

    mymsg := times:

    if (mymsg=-10) then error ERR_INTERRUPT end if:
    if (mymsg=-100) or (mymsg=-110) then error ERR_PRIME end if:
    if (mymsg=-200) then error ERR_FOURIERPRIME end if:
    if (times>10) then return 0 end if:

    ## end to use dims2 as a temp storage.
    for i from 2 to M+1 do
        dims2[i] := dims1[i]:
    end do:

    Res := Record('TIMES', 'ES', 'BOUNDS', 'DIMS1', 'DIMS2', 'ROOTS'):
    Res:-ES := es:         ## Fourier degrees that we need 
    Res:-BOUNDS := bounds: 
    Res:-DIMS1  := dims1:  ## TFT size + partial degrees
    Res:-DIMS2  := dims2:  ## TFT size + partial degrees
    Res:-ROOTS  := uroots: ## the prmitive roots (one in each dimension)
    Res:-TIMES  := times:
    return Res:
end proc;

#------------------------------------------------------------------------------
# Function: { TFTEvaluationM } 
# Briefly: { Evalute a polynomial at an array of points }
# Calling sequence:
# { TFTEvaluationM(N, M, in_rep, es, dims, ROOTS, p) }
# Input: 
# { N : index of the main variable }
# { M : number the variables that we specialize }
# { in_rep : Modpn polynomial }    
# { TFTPreRes : result from TFTPreComp }
# { p : Fourier prime }       
# Output : ECube
#------------------------------------------------------------------------------
TFTEvaluationM := proc(N, M, ININrep, es, dims, ROOTS, p)
    local Ecube, Edat, Esz, NMp, i, pSz, rep, INrep;
    
    ASSERT(M<=N, "M is at most N", M, N);

    INrep := PartialDegsAssert(ININrep):
    rep := CRepAssert(INrep, 1):
    pSz := calSizDgs(rep:-DGS, N):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N:
    NMp[2] := M:
    NMp[3] := p:

    ## dims[1] is 0, and not in use.
    Esz := mul(dims[i], i=2..N+1);

    if(Esz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Edat := Array(1..Esz, 'datatype' = 'integer[4]'):

    ConnectorModuleNew1:-TFTEvalM(NMp, es, dims, Esz, Edat, rep:-DGS, pSz, rep:-CRep:-COEF, ROOTS):

    Ecube := Record('Ees', 'Edims', 'Esize', 'Edata'):
    Ecube:-Ees := es;
    Ecube:-Edims := dims;
    Ecube:-Esize := Esz:
    ## size of Edat:
    Ecube:-Edata := Edat:
    ## data array, its first position is 1.
    return Ecube:
end proc:

#------------------------------------------------------------------------------
# Function: TFTInterpolationM
# Briefly:
# Calling sequence:
# {
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables appearing }
# { M : the number of variables evaluated, should be N-1 }
# { R : a modpn polynomial ring of characteristic `p` and with variable list }
# { ESTIDGS : esimate partial degrees for the output polynomial }
# { Edims, Esz, E : output of the TFT evaluation }
# Output:
# { a modpn multivariate polynomial in recden format }
#------------------------------------------------------------------------------
TFTInterpolationM := proc(R, N, M, ESTIDGS, Ecube, ROOTS, p)
local NMp, cVsz, coefVec, dVsz, pdegVec, rep;
    dVsz := estimatePartialDegVecSize(ESTIDGS, N) + 1:
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgs(ESTIDGS, N) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N: 
    NMp[2] := M: 
    NMp[3] := p:
    ConnectorModuleNew1:-TFTInterpM(NMp, Ecube:-Ees, Ecube:-Edims, Ecube:-Esize, Ecube:-Edata, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly := RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
    rep:-DGS := 0:
    rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF := 0:
    rep:-CRep:-PDGSV := pdegVec:
    rep:-CRep:-COEFV := coefVec:
    rep:-CRepCase := 2:
    return rep:
end proc:

#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables}
# { M : the number of variables to be evaluated, should be N-1 }
# { w : the minimum degree of input polynomials wrt the (common) main variable. }
# { bounds : a one-dimensional array containing (plus-one) bounds for
#   the resultant of the input polynomials w.r.t. their other
#   variables, that is, X_1, ..., X_M. Note the first slot is not used. }
# { Ecube1: the evaluation representation of the first polynomial. }
# { Ecube2: the evaluation representation of the second polynomial. }
# Output:
# { S : the sub-resultant chains of the input polynomials evaluated at each point of
#       their evaluation representation. }
# { Ssz : the total number of slots in 'S' }
# { Sdims : the dimensions of S, that is, 'bounds' followed by '[w, w]'.}
#------------------------------------------------------------------------------
GetResultantChainsTFT := proc(N, M, w, bounds, Ecube1, Ecube2, p)
local mymsg, Scube, Sdat, Ssz, i, ssdims;

    Ssz := mul(bounds[i], i=2..M+1);
    Ssz := Ssz*w*w:
    if (Ssz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    Sdat := Array(1..Ssz, 'datatype' = 'integer[4]'):
    mymsg := ConnectorModuleNew1:-GetResChainsTFT(N, M, w, Ssz, Sdat, 
                      Ecube1:-Edims, Ecube1:-Esize, Ecube1:-Edata, 
                      Ecube2:-Edims, Ecube2:-Esize, Ecube2:-Edata, p):

    if (mymsg=-10)  then error ERR_INTERRUPT: end if:
    if (mymsg=-100) or (mymsg=-110) then error ERR_PRIME: end if:
    if (mymsg=-200) then error ERR_FOURIERPRIME: end if:
    ssdims := Array(1..N+2, 'datatype'='integer[4]'):
    for i from 2 to N do ssdims[i] := bounds[i]: end do:

    ssdims[N+1] := w:
    ssdims[N+2] := w:
    Scube := Record('Ses', 'Sdims', 'Ssize', 'Sdata', 'SLc', 'SPoly'):
    Scube:-Ses := Ecube1:-Ees;
    Scube:-Sdims := ssdims:
    Scube:-Ssize := Ssz:
    Scube:-Sdata := Sdat:
    Scube:-SLc := Array(1..w):
    Scube:-SPoly := Array(1..w):
    return Scube:
end proc:

#------------------------------------------------------------------------------
# Function: { TFTInterpolatePolynomial }
# Briefly:
# Calling sequence:
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { ith: ith row in the subresultant chain. }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChainsTFT }
# { ROOTS : primitive roots used for evaluations }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the i-th subresultant }
#------------------------------------------------------------------------------
TFTInterpolatePolynomial := proc(R, N, M, ith, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep, slicedims, slicees, slicesz;

    if(((Scube:-SPoly)[ith+1]) <> 0) then return (Scube:-SPoly)[ith+1]: end if:

    slicees := Scube:-Ses:
    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] -1 end do:
    ESTIDGS[N+1] := w-1:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N: NMp[2] := M: NMp[3] := p:
    slicesz := iquo(Ssz, w):
    ConnectorModuleNew1:-TFTInterpithM(NMp, ith, w, slicesz, slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly := RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
    rep:-DGS := 0:
    rep:-CRep := 0:
    rep:-CRepCase :=-1:

    if(nargs=9) then
        rep := ReduceCoeffs(R, rep, args[9]):
    end if:
    (Scube:-SPoly)[ith+1] := rep:
    return rep:
end proc:
#------------------------------------------------------------------------------
# Function: { TFTInterpolateCoefficient }
# Briefly:
# Calling sequence:
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { ith: ith row in the subresultant chain. }
# { dth: dth column in the subresultant chain. }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChainsTFT }
# { ROOTS : primitive roots used for evaluations }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the coefficient in the i-th row and d-th column. }
#------------------------------------------------------------------------------
TFTInterpolateCoefficient := proc(R, N, M, ith, dth, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep,
    slicedims, slicees, slicesz, tmp, ww;

    slicees := Scube:-Ses:
    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] - 1 end do;
    ESTIDGS[N+1] := 0:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N:
    NMp[2] := M: 
    NMp[3] := p:
    ww := w*w;
    slicesz := iquo(Ssz, ww):
    tmp := slicedims[N+1]:
    slicedims[N+1] := 1:
    ConnectorModuleNew1:-TFTInterpithdthM(NMp, ith, dth, w, slicesz,
        slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):

    slicedims[N+1] := tmp:
    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly :=
        RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
    rep:-DGS := 0:
    rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF := 0:
    rep:-CRep:-PDGSV := pdegVec:
    rep:-CRep:-COEFV:=coefVec:
    rep:-CRepCase:=2:
    return rep:
end proc:

TFTInterpolateNextDefectiveLC := proc(R, N, M, start, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz,cVsz, coefVec, dVsz, i, j, pdegVec, rep, rep2,
    slicedims, slicees, slicesz, theend, tmp, ww, thestart;

    if start = w-1 then
        rep2 := Record('ith', 'LC');
        rep2:-ith := -1;
        rep2:-LC := 0:
        return rep2:
    end if:

    if (start<0) then thestart := 0: else thestart := start: end if:

    slicees := Scube:-Ses:
    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] -1 end do;
    ESTIDGS[N+1] := 0:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
    if(dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N: 
    NMp[2] := M: 
    NMp[3] := p:
    ww := w*w;
    slicesz := iquo(Ssz,ww):
    tmp := slicedims[N+1]:
    slicedims[N+1] := 1:
    theend := ConnectorModuleNew1:-TFTInterpNextDefectiveLCM(NMp, start, w,
        slicesz, slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    slicedims[N+1] := tmp:

    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    if theend = -1 then
        rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
        rep := 0;
        rep2 := Record('ith', 'LC');
        rep2:-ith := theend;
        rep2:-LC := rep:
    else
        rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
        rep:-RecdenPoly := RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
        rep:-DGS := 0:
        rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
        rep:-CRep:-COEF := 0:
        rep:-CRep:-PDGSV := pdegVec:
        rep:-CRep:-COEFV := coefVec:
        rep:-CRepCase := 2:
        rep2 := Record('ith', 'LC');
        rep2:-ith := theend;
        rep2:-LC := rep:
    end if:
    return rep2:
end proc:

TFTInterpolateNextLC:=proc(R, N, M, start, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz,cVsz, coefVec, dVsz, i, j, pdegVec, rep, rep2,
    slicedims, slicees, slicesz, theend, tmp, ww, thestart;

    if start = w-1 then
       rep2 := Record('ith', 'LC');
       rep2:-ith := -1;
       rep2:-LC := 0:
       return rep2:
    end if:

    if (start<0) then thestart := 0: else thestart := start: end if:
    for j from thestart+1 to w-1 do
        if(((Scube:-SLc)[j+1])<>0) then return (Scube:-SLc)[j+1]: end if:
    end do:

    slicees := Scube:-Ses:
    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] -1 end do;
    ESTIDGS[N+1] := 0:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N: 
    NMp[2] := M: 
    NMp[3] := p:
    ww := w*w;
    slicesz := iquo(Ssz, ww):
    tmp := slicedims[N+1]:
    slicedims[N+1] := 1:
    theend := ConnectorModuleNew1:-TFTInterpNextLCM(NMp, start, w, slicesz,
        slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    slicedims[N+1] := tmp:

    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    if theend = -1 then
        rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
        rep := 0;
        rep2 := Record('ith', 'LC');
        rep2:-ith := theend;
        rep2:-LC := rep:
    else
        rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
        rep:-RecdenPoly := RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
        rep:-DGS := 0:
        rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
        rep:-CRep:-COEF := 0:
        rep:-CRep:-PDGSV := pdegVec:
        rep:-CRep:-COEFV := coefVec:
        rep:-CRepCase := 2:
        rep2 := Record('ith', 'LC');
        rep2:-ith := theend;
        rep2:-LC := rep:
        (Scube:-SLc)[(rep2:-ith) + 1] := rep2:
    end if:
    return rep2:
end proc:

###############################################################################
##  Subproduct tree based functions, ALGEB version
###############################################################################
#------------------------------------------------------------------------------
# Function: { GetResultantChains_ALGEB }
# Briefly:
# Calling sequence:
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables appearing in `INrep` }
# { M : the number of variables to be evaluated }
# { w : the minimum degree of the input polynomials w.r.t. their
#       (common) main variable. }
# { bounds : a one-dimensional array containing (plus-one) bounds for the
#            resultant of the input polynomials w.r.t. their other variables,
#            that is, X_1, ..., X_M. Note the first slot is not used. }
# { Edims1, E1Sz, E1: the evaluation representation of the first polynomial. }
# { Edims2, E2Sz, E2: the evaluation representation of the second polynomial. }
# Output:
# { S : the sub-resultant chains of the input polynomials evaluated at
#       each point of their evaluation representation. }
# { Ssz : the total number of slots in 'S' }
# { Sdims : the dimensions of S, that is, 'bounds' followed by '[w,w]'.}
#------------------------------------------------------------------------------
GetResultantChains_ALGEB := proc(N, M, w, bounds, Ecube1, Ecube2, p)
    local mymsg, Scube, Sdat, Ssz, i, ssdims;

    Ssz := mul(bounds[i], i=2..M+1);
    Ssz := Ssz*w*w:
    if (Ssz > MEMORY_BOUND) then error ERR_MEMORY end if;

    Sdat := MallocGcArray(Ssz);
    ASSERT(Sdat<>0, "NULL MaplePointer not expected");
    mymsg := ConnectorModuleNew1:-GetResChains_ALGEB(N, w, Ssz, Sdat,
        Ecube1:-Edims, Ecube1:-Esize, Ecube1:-Edata, Ecube2:-Edims,
        Ecube2:-Esize, Ecube2:-Edata, p);

    if (mymsg=-10)  then error  ERR_INTERRUPT end if;
    if (mymsg=-100 or mymsg=-110) then error ERR_PRIME end if;
    if (mymsg=-200) then error ERR_FOURIERPRIME end if;
    ssdims := Array(1..N+2, 'datatype'='integer[4]'):
    for i from 2 to N do ssdims[i] := bounds[i]; end do;
    ssdims[N+1] := w;
    ssdims[N+2] := w;
    Scube := Record('Sdims', 'Ssize', 'Sdata', 'SLc', 'SPoly');
    Scube:-Sdims := ssdims;
    Scube:-Ssize := Ssz;
    Scube:-Sdata := Sdat;
    Scube:-SLc := Array(1..w);
    ## buffer space for the initials of the subresultants
    Scube:-SPoly := Array(1..w);
    ## buffer space for the subresultants
    return Scube;
end proc:
#------------------------------------------------------------------------------
# Function: { FastEvaluationM1_ALGEB }
# Briefly:
# Calling sequence:
# Input:
# { INrep :  a multivariate modpn polynomial }
# { p : prime number, the characteristic of the field }
# { N : the total number of variables appearing in `INrep` }
# { M : the number of variables to be evaluated }
# { PSTREE : the data-structure holding the points and the sub-product
#            trees needed to evaluate `INrep` }
# Output:
# { E : an evaluation representation of the `INrep` where the `M`
#       lowest variables are evaluated }
# { Esz : an integer, the total number of slots of `E` }
# { PTSTREE:-PTdims1 : a one-dimensional array of size `N+1` where
#   the first slot is not used and the others give the diemnsions of `E`
#   from the smallest to the largest variable in `INrep`
#   E.g.  (PTSTREE:-PTdims1)[2] is the first  dimension of 'E' .}
#------------------------------------------------------------------------------
FastEvaluationM1_ALGEB := proc(N, M, ININrep, PTSTREE, p)
    local mymsg, Ecube, Edat, Esz, NMp, PHWD, i, pSz, rep, INrep;

    INrep := PartialDegsAssert(ININrep):
    rep := CRepAssert(INrep, 1):
    pSz := calSizDgs(rep:-DGS,N):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N: 
    NMp[2] := M: 
    NMp[3] := p:
    PHWD := Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1] := PTSTREE:-PTptssSz: 
    PHWD[2] := PTSTREE:-PThsSz: 
    PHWD[3] := PTSTREE:-PTWNBsSz: 
    PHWD[4] := PTSTREE:-PTdatasSz:
    
    Esz := mul(PTSTREE:-PTdims1[i], i=2..N+1);
    if (Esz > MEMORY_BOUND) then error ERR_MEMORY end if:
    Edat := MallocGcArray(Esz);
    ASSERT(Edat<>0, "NULL MaplePointer not expected");

    mymsg := ConnectorModuleNew1:-FastEvalM_ALGEB(NMp, PHWD, PTSTREE:-PTbounds,
        PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs, PTSTREE:-PTNoNodess,
        PTSTREE:-PTBasess, PTSTREE:-PTdatas, PTSTREE:-PTdims1, Esz,
        Edat, rep:-DGS, pSz, rep:-CRep:-COEF):

    if (mymsg=-10) then error ERR_INTERRUPT end if:
    if (mymsg=-100 or mymsg=-110) then error ERR_PRIME end if:
    if (mymsg=-200) then error ERR_FOURIERPRIME end if:
    Ecube := Record('Edims', 'Esize', 'Edata'):
    Ecube:-Edims := PTSTREE:-PTdims1:
    Ecube:-Esize := Esz:
    Ecube:-Edata := Edat:
    return Ecube:
end proc:

## Ecube for the second polynomial.
FastEvaluationM2_ALGEB := proc(N, M, ININrep, PTSTREE, p)
    local mymsg, Ecube, Edat, Esz, NMp, PHWD, i, pSz, rep, INrep;

    INrep := PartialDegsAssert(ININrep):
    rep := CRepAssert(INrep, 1):
    pSz := calSizDgs(rep:-DGS, N):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N: 
    NMp[2] := M: 
    NMp[3] := p:
    PHWD := Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1] := PTSTREE:-PTptssSz: 
    PHWD[2] := PTSTREE:-PThsSz: 
    PHWD[3] := PTSTREE:-PTWNBsSz: 
    PHWD[4] := PTSTREE:-PTdatasSz:

    Esz := mul(PTSTREE:-PTdims2[i], i=2..N+1);
    if (Esz > MEMORY_BOUND) then error ERR_MEMORY end if:
    Edat := MallocGcArray(Esz);
    ASSERT(Edat<>0, "NULL MaplePointer not expected");

    mymsg := ConnectorModuleNew1:-FastEvalM_ALGEB(NMp, PHWD, PTSTREE:-PTbounds,
        PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs, PTSTREE:-PTNoNodess,
        PTSTREE:-PTBasess, PTSTREE:-PTdatas, PTSTREE:-PTdims2, Esz,
        Edat, rep:-DGS, pSz, rep:-CRep:-COEF):

    if (mymsg=-10) then error ERR_INTERRUPT end if:
    if (mymsg=-100 or mymsg=-110) then error ERR_PRIME end if:
    if (mymsg=-200) then error ERR_FOURIERPRIME end if:
    Ecube := Record('Edims', 'Esize', 'Edata'):
    Ecube:-Edims := PTSTREE:-PTdims2:
    Ecube:-Esize := Esz:
    Ecube:-Edata := Edat:
    return Ecube:
end proc:
#------------------------------------------------------------------------------
# Function: { FastInterpolationM_ALGEB }
# Briefly:
# Calling sequence:
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables appearing }
# { M : the number of variables to be evaluated }
# { R : a modpn polynomial ring of characteristic `p` and with variable list }
# { PTSTREE : sub-product tree data-structure. }
# { ESTIDGS : esimate partial degrees for the output polynomial }
# { Edims, Esz, E : output of the evaluation }
# Output:
# { a modpn multivariate polynomial in recden format }
#------------------------------------------------------------------------------
FastInterpolationM_ALGEB := proc(R, N, M, PTSTREE, ESTIDGS, Ecube, p)
    local mymsg, E, Esz, NMp, PHWD, cVsz, coefVec, dVsz, edims, pdegVec, rep;

    edims := Ecube:-Edims:
    Esz := Ecube:-Esize:
    E := Ecube:-Edata:
    ASSERT(E<>0, "NULL MaplePointer not expected");

    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N: 
    NMp[2] := M: 
    NMp[3] := p:
    PHWD := Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1] := PTSTREE:-PTptssSz: 
    PHWD[2] := PTSTREE:-PThsSz: 
    PHWD[3] := PTSTREE:-PTWNBsSz: 
    PHWD[4] := PTSTREE:-PTdatasSz:
    
    mymsg := ConnectorModuleNew1:-FastInterpM_ALGEB(NMp, PHWD, PTSTREE:-PTbounds,
        PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs,  PTSTREE:-PTNoNodess, 
        PTSTREE:-PTBasess, PTSTREE:-PTdatas, edims, Esz, E, dVsz, pdegVec,
        cVsz, coefVec):

    if (mymsg=-10)  then error ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error ERR_PRIME: end if:
    if (mymsg=-200) then error ERR_FOURIERPRIME: end if:
    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly := 
        RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
    rep:-DGS := 0:
    rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF := 0:
    rep:-CRep:-PDGSV:=pdegVec:
    rep:-CRep:-COEFV:=coefVec:
    rep:-CRepCase := 2:
    return rep:
end proc:
#------------------------------------------------------------------------------
# Function: { FastInterpolatePolynomial_ALGEB }
# Briefly:
# { interpolate the ith subresultant from the Scube }
# Calling sequence:
# { R : modpn polynomial ring }
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { ith: ith row in the subresultant chain. }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChains_ALGEB }
# { PTSTREE: subproduct tree }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the i-th subresultant }
#------------------------------------------------------------------------------
FastInterpolatePolynomial_ALGEB := proc(R, N, M, ith, PTSTREE, w, Scube, p)
    local ESTIDGS, NMp, PHWD, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep,
    slicedims, slicesz;

    if (((Scube:-SPoly)[ith+1]) <> 0) then
         return (Scube:-SPoly)[ith+1]:
    end if:

    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ASSERT(S<>0, "NULL MaplePointer not expected");

    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := PTSTREE:-PTbounds[i] - 1 end do:
    ESTIDGS[N+1] := w-1:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N:
    NMp[2] := M:
    NMp[3] := p:
    PHWD := Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1] := PTSTREE:-PTptssSz:
    PHWD[2] := PTSTREE:-PThsSz:
    PHWD[3] := PTSTREE:-PTWNBsSz:
    PHWD[4] := PTSTREE:-PTdatasSz:
    slicesz := iquo(Ssz,w):
    ConnectorModuleNew1:-FastInterpithM_ALGEB(NMp, PHWD, ith, PTSTREE:-PTbounds,
        PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs, PTSTREE:-PTNoNodess,
        PTSTREE:-PTBasess, PTSTREE:-PTdatas, w, slicesz, slicedims, Ssz,
        S, dVsz, pdegVec, cVsz, coefVec):

    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly := 
        RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):

    rep:-DGS := 0:
    rep:-CRep := 0:
    rep:-CRepCase := -1:

    if (nargs=9) then rep := ReduceCoeffs(R, rep, args[9]): end if:
    (Scube:-SPoly)[ith+1] := rep:
    return rep:
end proc:

#------------------------------------------------------------------------------
# Function: { FastInterpolateCoefficient_ALGEB } 
# Briefly: { Interpolate a coefficient from the subresultant chain. }
# Calling sequence:
# Input:
# N: { the number of variables from the least one to the main one. }
# M: { the number of variables has been evaluated. }
# ith: { ith row in the subresultant chain. }
# dth: { dth column in the subresultant chain }
# PTSTREE: { the points and subproduct-tree }
# w : { degree(poly2)-1. }
# Output:
# { modpn polynomial, the coefficient in the i-th row and d-th column. }
#------------------------------------------------------------------------------
FastInterpolateCoefficient_ALGEB := proc(R, N, M, ith, dth, PTSTREE, w, Scube, p)
    local ESTIDGS, NMp, PHWD, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep,
    slicedims, slicesz, tmp, ww;

    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ASSERT(S<>0, "NULL MaplePointer not expected");
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := PTSTREE:-PTbounds[i] - 1 end do;
    ESTIDGS[N+1] := 0:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1) + 1:
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N:
    NMp[2] := M:
    NMp[3] := p:
    PHWD := Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1] := PTSTREE:-PTptssSz:
    PHWD[2] := PTSTREE:-PThsSz:
    PHWD[3] := PTSTREE:-PTWNBsSz:
    PHWD[4] := PTSTREE:-PTdatasSz:
    ww := w*w;
    slicesz := iquo(Ssz, ww):
    tmp := slicedims[N+1]:
    slicedims[N+1] := 1:

    ConnectorModuleNew1:-FastInterpithdthM_ALGEB(NMp, PHWD, ith, dth,
        PTSTREE:-PTbounds, PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs,
        PTSTREE:-PTNoNodess, PTSTREE:-PTBasess, PTSTREE:-PTdatas, w,
        slicesz, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec):

    slicedims[N+1] := tmp:
    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly :=
        RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
    rep:-DGS := 0:
    rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF := 0:
    rep:-CRep:-PDGSV := pdegVec:
    rep:-CRep:-COEFV := coefVec:
    rep:-CRepCase := 2:
    return rep:
end proc:
#------------------------------------------------------------------------------
# Function: { FastInterpolateNextDefectiveLC_ALGEB }
# Briefly: { Interpolate the next defective leading coefficient in a SCUBE }
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { start: the row (or index) at which to start seaching a subresultant
# i        whose LC is not zero, going bottom-up }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChains_ALGEB }
# { PTSTREE: subproduct tree }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the next defective leading coefficient from
#   the subresultant start }
#------------------------------------------------------------------------------
FastInterpolateNextDefectiveLC_ALGEB := proc(R, N, M, start, PTSTREE, w, Scube, p)
    local ESTIDGS, NMp, PHWD, S, Ssz, cVsz, coefVec, dVsz, i, j, pdegVec, rep,
    rep2, slicedims, slicesz, theend, tmp, ww, thestart;

    if (start = w-1) then
       rep2:=Record('ith', 'LC');
       rep2:-ith:=-1;
       rep2:-LC:=0:
       return rep2:
    end if:

    if(start<0) then thestart := 0: else thestart := start: end if:

    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ASSERT(S<>0, "NULL MaplePoitner not expected");
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := PTSTREE:-PTbounds[i] - 1 end do;
    ESTIDGS[N+1] := 0:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1) + 1:
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N:
    NMp[2] := M:
    NMp[3] := p:
    PHWD := Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1] := PTSTREE:-PTptssSz: PHWD[2] := PTSTREE:-PThsSz:
    PHWD[3] := PTSTREE:-PTWNBsSz: PHWD[4] := PTSTREE:-PTdatasSz:
    ww := w * w;
    slicesz := iquo(Ssz, ww):
    tmp := slicedims[N+1]:
    slicedims[N+1] := 1:

    theend := ConnectorModuleNew1:-FastInterpNextDefectiveLCM_ALGEB(NMp, PHWD,
        start, PTSTREE:-PTbounds, PTSTREE:-PTptss, PTSTREE:-PThs,
        PTSTREE:-PTWs, PTSTREE:-PTNoNodess, PTSTREE:-PTBasess, PTSTREE:-PTdatas,
        w, slicesz, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec):

    slicedims[N+1] := tmp:
    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    if theend = -1 then
      rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep := 0;
      rep2 := Record('ith', 'LC');
      rep2:-ith := theend;
      rep2:-LC := rep:
    else
      rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:-RecdenPoly :=
        RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
      rep:-DGS := 0:
      rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
      rep:-CRep:-COEF := 0:
      rep:-CRep:-PDGSV := pdegVec:
      rep:-CRep:-COEFV := coefVec:
      rep:-CRepCase := 2:
      rep2 := Record('ith', 'LC');
      rep2:-ith := theend;
      rep2:-LC := rep:
    end if:
    return rep2:
end proc:

#------------------------------------------------------------------------------
# Function: { FastInterpolateNextLC_ALGEB }
# Briefly: { Interpolate the next leading coefficient in a SCUBE }
# Input:
# { R : modpn polynomial ring}
# { N : the total number of variables }
# { M : the number of variables to be evaluated, should be N-1 }
# { start: index of a subresultant }
# { w : minimum of the main degrees }
# { Scube : data structure built by GetResultantChains_ALGEB }
# { PTSTREE : subproduct tree }
# { p : Fourier prime }
# Output :
# { modpn polynomial, the next leading coefficient from the subresultant start }
#------------------------------------------------------------------------------
FastInterpolateNextLC_ALGEB := proc(R, N, M, start, PTSTREE, w, Scube, p)
    local ESTIDGS, NMp, PHWD, S, Ssz, cVsz, coefVec, dVsz, i, j, pdegVec,
    rep, rep2, slicedims, slicesz, theend, tmp, ww, thestart;

    if (start = w-1) then
       rep2 := Record('ith', 'LC');
       rep2:-ith := -1;
       rep2:-LC :=0:
       return rep2:
    end if:
    
    if (start<0) then thestart := 0: else thestart := start: end if:
    
    for j from thestart+1 to w-1 do
       if(((Scube:-SLc)[j+1])<>0) then
          return (Scube:-SLc)[j+1]:
       end if:
    end do:
    
    slicedims := Scube:-Sdims:
    Ssz := Scube:-Ssize:
    S := Scube:-Sdata:
    ASSERT(S<>0, "NULL MaplePointer not expected");
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := PTSTREE:-PTbounds[i] - 1 end do;
    ESTIDGS[N+1] := 0:
    dVsz := estimatePartialDegVecSizefrom2(ESTIDGS, N+1) + 1:
    if (dVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    pdegVec := Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz := calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if (cVsz > MEMORY_BOUND) then error ERR_MEMORY: end if:
    coefVec := Array(1..cVsz, 'datatype'='integer[4]'):
    NMp := Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1] := N:
    NMp[2] := M:
    NMp[3] := p:
    PHWD := Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1] := PTSTREE:-PTptssSz: 
    PHWD[2] := PTSTREE:-PThsSz:
    PHWD[3] := PTSTREE:-PTWNBsSz: 
    PHWD[4] := PTSTREE:-PTdatasSz:
    ww := w * w;
    slicesz := iquo(Ssz, ww):
    tmp := slicedims[N+1]:
    slicedims[N+1] := 1:
    
    theend := ConnectorModuleNew1:-FastInterpNextLCM_ALGEB(NMp, PHWD, start,
        PTSTREE:-PTbounds, PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs,
        PTSTREE:-PTNoNodess, PTSTREE:-PTBasess, PTSTREE:-PTdatas, w, slicesz,
        slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec):

    slicedims[N+1] := tmp:
    rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    if theend = -1 then
      rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep := 0;
      rep2 := Record('ith', 'LC');
      rep2:-ith := theend;
      rep2:-LC := rep:
    else
      rep := Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:-RecdenPoly :=
        RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
      rep:-DGS := 0:
      rep:-CRep := Record('COEF', 'PDGSV', 'COEFV'):
      rep:-CRep:-COEF := 0:
      rep:-CRep:-PDGSV := pdegVec:
      rep:-CRep:-COEFV := coefVec:
      rep:-CRepCase := 2:
      rep2 := Record('ith', 'LC');
      rep2:-ith := theend;
      rep2:-LC := rep:
      (Scube:-SLc)[(rep2:-ith)+1] := rep2:
    end if:
    return rep2:
end proc:

#------------------------------------------------------------------------------
# Function:
# { PolynomialRing }
# Briefly:
# { Create a multivariate modpn ring.}
# Calling sequence:
# { PolynomialRing(p, vars)}
# Input:
# { p : a modpn prime}
# { vars : list of variables }
# Output:
# { A modpn ring, reprsented by a record. }
#------------------------------------------------------------------------------
PolynomialRing := proc(p,vars)
    local R;
    R := Record('prime', 'VarList');
    R:-prime := p;
    R:-VarList := vars;
    return R;
end proc;
#------------------------------------------------------------------------------
# Function:
#     {getDGS}
# Briefly:
#     {Return the partial degree vector of an input multivariate
#      modpn polynomial.}
# Calling sequence:
#     {getDGS (InRec)}
# Input:
#     {InRec: A mutlivariate modpn polynomial.}
# Output:
#     {A paritial degrees vector of 'InRec'. }
#------------------------------------------------------------------------------
getDGS := proc (InRec)
    local Rec;
    Rec:=PartialDegsAssert(InRec):
    return Rec:-DGS:
end proc:


#------------------------------------------------------------------------------
# Function:
# {PartialDeg}
# Briefly:
# {Return the partial degree of an input modpn polynomial in the give variable.}
# Calling sequence:
# { PartialDeg(R, InRec, v) }
# Input:
# {R: A modpn ring}
# {InRec: modpn polynomial}
# {v: variable}
# Output:
# {The paritial degree of 'InRec' in 'v'.}
#------------------------------------------------------------------------------
PartialDeg := proc (R, InRec, v)
    local i, deg, no, Rec;
    deg:=0:
    no:=nops(R:-VarList):
    Rec:=PartialDegsAssert(InRec):
    for i from 1 to no do
       if ((R:-VarList)[i] = v) then
          deg:=Rec:-DGS[no-i+1]:
          break:
       end if:
    end do:
    return deg:
end proc:

#------------------------------------------------------------------------------
# Function:
# {PartialDegsAssert}
# Briefly:
# { To enforce the computation of the partial degrees of the
# input multivariate modpn polynomial and to those in the
# representation of that polynomial.If the partial degrees vector
# exist already then does nothing, else calculate the partial degrees.}
# Calling sequence:
# { PartialDegsAssert(Rec) }
# Input:
# { Rec: a multivariate modpn polynomial. }
# Output:
# { The input multivariate modpn polynomial with partial degree vector avaiable.}
#------------------------------------------------------------------------------
PartialDegsAssert := proc (Rec)
  local NewRec:
  if (Rec:-DGS = 0) then
      NewRec:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      NewRec:-DGS:=(RecdenConnector:-PartialDegs(Rec:-RecdenPoly))[2]:
      NewRec:-RecdenPoly:=Rec:-RecdenPoly:
      NewRec:-CRep:=Rec:-CRep:
      NewRec:-CRepCase:=Rec:-CRepCase:
  else
      NewRec:=Rec:
  end if:
  return NewRec:
end proc:

#------------------------------------------------------------------------------
# Function:
# { PolynomialConvertIn }
# Briefly:
# { Converting a Maple DAG polynomial into a modpn polynomial.}
# Calling sequence:)
# { PolynomialConvertIn(R, poly) }
# Input:
# { R : modpn polynomial ring }
# { poly : Maple DAG polynomial }
# Output:
# { Returns a modpn multivariate polynomial, represented by the list of its
#   partial degrees, its Recden format, one of each C representation
#   and the case for its C representations.}
#------------------------------------------------------------------------------
PolynomialConvertIn:=proc(R, poly)
   local RPoly, Rec:
   RPoly := Algebraic:-RecursiveDensePolynomials:-rpoly(poly, R:-VarList):
   RPoly := Algebraic:-RecursiveDensePolynomials:-modrpoly(RPoly, R:-prime);
   Rec:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
   #Rec:-DGS:=(RecdenConnector:-PartialDegs(RPoly))[2]:
   Rec:-DGS:=0:
   Rec:-RecdenPoly:=RPoly:
   Rec:-CRep:=0:
   Rec:-CRepCase:=-1:
   return Rec:
end proc;

# GetCoefficient:=proc(INrep)
#    local Rec, newrpoly;
#    #kernelopts(opaquemodules = false):
#    newrpoly:=Algebraic:-RecursiveDensePolynomials:-coeffrpoly(INrep:-RecdenPoly,Algebraic:-RecursiveDensePolynomials:-degrpoly(INrep:-RecdenPoly));
#    Rec:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
#    #Rec:-DGS:=(RecdenConnector:-PartialDegs(newrpoly))[2]:
#    Rec:-DGS:=0:
#    Rec:-RecdenPoly:=newrpoly:
#    Rec:-CRep:=0:
#    Rec:-CRepCase:=-1:
#    Rec:
# end proc;

#------------------------------------------------------------------------------
# Function:
#    {Rpoly2Modpn}
# Briefly:
#    {Converting a Recden polynomial into a modpn polynomial}
# Calling sequence:
#    { Rpoly2Modpn(R, RPoly) }
# Input:
#    { R: a modpn ring}
#    { RPoly: a Recden polynomial}
# Output:
#    { A modpn polynomial.}
#------------------------------------------------------------------------------
Rpoly2Modpn:=proc(R, RPoly)
   local Rec;
   Rec:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
   #Rec:-DGS:=(RecdenConnector:-PartialDegs(RPoly))[2]:
   Rec:-DGS:=0:
   Rec:-RecdenPoly:=RPoly:
   Rec:-CRep:=0:
   Rec:-CRepCase:=-1: ## Means that no C representation is available
                      ## 1 would mean the Cube representation
                      ## 2 would mean the 2-vector representation
   return Rec:
end proc:

#------------------------------------------------------------------------------
# Function:
#   { PolynomialConvertInUNI}
# Briefly:
#   {Converting a univariate Maple DAG polynomial into a univariate modpn polynomial.
#    (univariate modpn polynomial is different from multivariate modpn polynomial)}
# Calling sequence:
#   {PolynomialConvertInUNI(p, poly, var)}
# Input:
#   { p : prime number }
#   { poly : Maple DAG polynomial (must be univariate) }
#   { var : the symbol for the variable of the univariate polynomial. }
# Output:
#   { Returns a modpn univariate polynomial, represented by Recden
#   format and a C representation (consisting of a degree and a
#    coefficient array. }
#------------------------------------------------------------------------------
PolynomialConvertInUNI:=proc(p, poly, var)
   local RPoly, Rec;
   RPoly := Algebraic:-RecursiveDensePolynomials:-rpoly(poly, [var]):
   RPoly := Algebraic:-RecursiveDensePolynomials:-modrpoly(RPoly, p):
   Rec:=Record('RecdenPoly', 'CRep'):
   Rec:-RecdenPoly:=RPoly:
   Rec:-CRep:=0:
   return Rec:
end proc:

#------------------------------------------------------------------------------
# Function:
#    {PolynomialConvertOut}
# Briefly:
#    {Converting a modpn polynomial to a Maple DAG polynomial.}
# Calling sequence:
#    {PolynomialConvertOut(rep)}
# Input:
#    { rep : A multivariate modpn polynomial }
# Output:
#    { A Maple DAG polynomial. }
#------------------------------------------------------------------------------
PolynomialConvertOut:=proc(rep)
  local res, p:
  if(rep=0) then
    return 0;
  end if:
  res:=Algebraic:-RecursiveDensePolynomials:-rpoly(rep:-RecdenPoly);
  #p:= op(rep:-RecdenPoly)[1][1]:
  #Expand(res) mod p;
  #expand(res):
  return res;
end proc;

#------------------------------------------------------------------------------
# Function:
#   {PolynomialConvertOutUNI}
# Briefly:
#   {Converting a univariate modpn polynomial to a Maple DAG polynomial.}
# Calling sequence:
#   {PolynomialConvertOutUNI(rep)}
# Input:
#   { rep: a univariate modpn polynomial.}
# Output:
#   { A Maple DAG polynomial. }
#------------------------------------------------------------------------------
PolynomialConvertOutUNI:=proc(rep)
    return Algebraic:-RecursiveDensePolynomials:-rpoly(rep:-RecdenPoly);
end proc;

#------------------------------------------------------------------------------
# Function:
#   {CRepAssert}
# Briefly:
#   {To assert the C encoding in the modpn polynomial is avaiable}
# Calling sequence:
#   {CRepAssert (rep, case)}
# Input:
#   { rep : a multivariate modpn polynomial }
#   { case : a type of C representation }
#   { For `case` the values '0', '1', and '2' correspond respectively
#     to no C representations, C-Cube, 2-Vector.
# Output:
#   { If `rep` does not have that case representation, then returns a
#   copy of `rep` which has it, otherwise returns `rep` itself. }
#------------------------------------------------------------------------------
CRepAssert := proc(rep, case)
  local newrep, rec, resrep;
  resrep:=rep:
  if rep:-CRep = 0 or rep:-CRepCase <> case then
      newrep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      newrep:-DGS:=rep:-DGS:
      newrep:-RecdenPoly:=rep:-RecdenPoly:
      newrep:-CRepCase:=case:
      # cut-off needed. Now assume the second case.
      # If newrep:-CRepCase = 1. this means we want CUBE
      # data rep. So the record will looks like
      #  (COEF=STH, PDGSV=0, COEFV=0)
      # if newrep:-CRepCase = 2, this means
      # this means we want compact C format (2V)
      #   this implies (COEF=0, PDGSV=STH, COEFV=STH)

      if newrep:-CRepCase = 1 then
         newrep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
         newrep:-CRep:-COEF:=RecdenConnector:-RecdenPoly2CData(rep:-RecdenPoly):
         newrep:-CRep:-PDGSV:=0:
         newrep:-CRep:-COEFV:=0:
      end if:
      if newrep:-CRepCase =2 then
         newrep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
         newrep:-CRep:-COEF:=0:
         rec:=RecdenConnector:-RecdenPoly2C2V(rep:-RecdenPoly):
         newrep:-CRep:-PDGSV:=rec[1]:
         newrep:-CRep:-COEFV:=rec[2]:
      end if:
      resrep:=newrep:
  end if:
  return resrep:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {CRepAssertUNI}
# Briefly:
#   {To assert the C encoding available for the univariate modpn. }
# Calling sequence:
#   { CRepAssertUNI(rep) }
# Input:
#   { rep : a modpn polynomial (assuming that it is
#    "mathematically" univariate}
# Output:
#   { if `rep` does have a C representation, then returns a copy of
#     `rep` as a univariate (=UNI) modpn polynomial. }
#------------------------------------------------------------------------------
CRepAssertUNI := proc(rep)
  local newrep, rec, resrep;
  resrep:=rep:
  if rep:-CRep = 0 then
      newrep:=Record('RecdenPoly', 'CRep'):
      newrep:-RecdenPoly:=rep:-RecdenPoly:
      newrep:-CRep:=Record('DEG', 'COEFV'):
      rec:=RecdenConnector:-RecdenPoly2CUni(newrep:-RecdenPoly):
      newrep:-CRep:-DEG:=rec[1]:
      newrep:-CRep:-COEFV:=rec[2]:
      resrep:=newrep:
  end if:
  return resrep:
end proc:

#------------------------------------------------------------------------------
# Function:
#   { shiftArrR }
# Briefly:
#   { To shift an vector 'A' right (towards to the bigger direction) m slots. }
# Calling sequence:
#   {shiftArrR(A, n, m)}
# Input:
#   {A : A Maple vector. }
#   {n : The size of 'A'.}
#   {m : The number of slots to shift.}
# Output:
#   {The shifted vector.}
#-----------------------------------------------------------------------------
# shift right by m of a A with size n.
shiftArrR:= proc (A, n, m)
   local B, i, s;
   s:=n + m:
   if(s>MEMORY_BOUND) then error ERR_MEMORY: end if:
   B:=Array(1..s, 'datatype' = 'integer[4]'):
   for i from 1 to n do
     B[i+m]:=A[i]:
   end do:
   return B:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {shiftArrL}
# Briefly:
#   {To shift an vector 'A' left (towards to the smaller direction) m slots.}
# Calling sequence:
#   {shiftArrL(A, n, m)}
# Input:
#   {A : a vector.}
#   {n : the size of vector 'A'.}
#   {m : the number of slots to shift.}
# Output:
#   {The shifted vector.}
#-----------------------------------------------------------------------------
shiftArrL:= proc (A, n, m)
   local B, i, s;
   s:=n - m:
   if (s>MEMORY_BOUND) then error ERR_MEMORY: end if:
   B:=Array(1..s, 'datatype' = 'integer[4]'):
   for i from 1 to s do
     B[i]:=A[i+m]:
   end do:
   return B:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {addArr}
# Briefly:
#   {Adding two vectors}
# Calling sequence:
#   {addArr(arr1, arr2, n)}
# Input:
#   { arr1 : an array of C ints }
#   { arr2 : an array of C ints }
#   { n : the length of these arrays }
# Output:
#   { The sum of these arrays regarded as vectors. }
#------------------------------------------------------------------------------
# add two arrays with their first n elems.
addArr := proc (arr1, arr2, n)
   local  arrr, i;
   if(n>MEMORY_BOUND) then error ERR_MEMORY; end if:
   arrr:=Array(1..n, 'datatype' = 'integer[4]'):
   for i from 1 to n do
     arrr[i]:=arr1[i]+arr2[i]:
   end do:
   return arrr:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {calSizDgs}
# Briefly:
#   {Calculate the size of the coefficients vector with respect to the degree vector
#    The size equals to "(d1+1)(d2+1)...(dn+1)" }
# Calling sequence:
#   {calSizDgs(dgs, n)}
# Input:
#   { dgs : an array of partial degrees of some polynomial }
#   { n : the number of variables }
# Output:
#   { The size of the C cube representation }
#------------------------------------------------------------------------------
calSizDgs:=proc(dgs, n)
  local i, sz;
  sz:=1:
  for i from 1 to n do
     sz:=sz*(dgs[i]+1):
  end do:
  sz:
end proc:

# a little variant from above one.
# where  dgs[2]=d1, ..., dgs[i+1]=di, ..., dgs[n+1]=dn.
calSizDgsfrom2:=proc(dgs, n)
  local i, sz;
  sz:=1:
  for i from 2 to n do
     sz:=sz*(dgs[i]+1):
  end do:
  return sz:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {estimatePartialDegVecSize}
# Briefly:
#   {estimate the size of buffer for the coefficents vector in 2-VECTOR representation. }
# Calling sequence:
#   {}
# Input:
#   { dgs : an array of partial degrees of some polynomial }
#   { n : the number of variables }
# Output:
#   { The size of the "degree tree" in the two-vector C representation. }
#------------------------------------------------------------------------------
estimatePartialDegVecSize := proc(dgs, n)
   local ac, i, siz;
   siz:=1:
   ac:=1:
   for i from n by -1 to 1 do
       ac:=(dgs[i]+1)*ac:
       siz:=siz+ac:
   end do:
   # the first slot keeps the size of vector.
   siz+1:
end proc:

# The same as the above one, except that the data in dgs starts from
# the second slot.
estimatePartialDegVecSizefrom2:= proc(dgs, n)
   local ac, i, siz;
   siz:=1:
   ac:=1:
   for i from n by -1 to 2 do
       ac:=(dgs[i]+1)*ac:
       siz:=siz+ac:
   end do:
   # the first slot keeps the size of vector.
   return siz+1:
end proc:

#============================================================#
#                Multi-variate TFTFFT                        #
#============================================================#
# TFTFFT multivariate multiplication.
#------------------------------------------------------------------------------
# Function:
#    {TFTFFTMul}
# Briefly:
#    {Multivariate polynomial based on FFT or TFT.}
# Calling sequence:
#    {TFTFFTMul(R, ININrep1, ININrep2, case, p)}
# Input:
#    { R : a modpn polynomial ring }
#    { INrep1 : a modpn polynomial }
#    { INrep2 : a modpn polynomial }
#    { case : type of C representation for the output }
# Output:
#    { The product of `INrep1` and 'INrep2` }
#------------------------------------------------------------------------------
TFTFFTMul := proc (R, ININrep1, ININrep2, case, p)
    local cvbuf, cvsz, dgsr, n, p1bufsz, p2bufsz, pvbuf, pvsz, rbuf, rbufsz,
    rep, rep1, rep2, INrep1, INrep2;

    #to assert two input dgs have the same variables.
    INrep1:=PartialDegsAssert(ININrep1):
    INrep2:=PartialDegsAssert(ININrep2):
    n:=ArrayNumElems(INrep1:-DGS):
    dgsr:=addArr(INrep1:-DGS, INrep2:-DGS, n):
    p1bufsz:=calSizDgs(INrep1:-DGS,n):
    p2bufsz:=calSizDgs(INrep2:-DGS,n):
    rbufsz:=calSizDgs(dgsr,n):

    #if rbufsz>100 then case:=2 else case:=1 end if:

    rep1:=CRepAssert(INrep1, case):
    rep2:=CRepAssert(INrep2, case):
    if(rbufsz>MEMORY_BOUND) then error ERR_MEMORY; end if:
    rbuf:=Array(1..rbufsz, 'datatype' = 'integer[4]'):
    pvsz:=estimatePartialDegVecSize(dgsr,n,1):
    if(pvsz>MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pvbuf:=Array(1..pvsz, 'datatype' = 'integer[4]'):
    # the first slot of cvsz keeps the size of vector.
    cvsz:=rbufsz+1:
    if(cvsz>MEMORY_BOUND)  then error ERR_MEMORY: end if:
    cvbuf:=Array(1..cvsz, 'datatype' = 'integer[4]'):
    rep:=0:

    if case=1 then
        ConnectorModule:-MTFTFFT(n, dgsr, rbufsz, rbuf, rep1:-DGS, p1bufsz,
            rep1:-CRep:-COEF, rep2:-DGS, p2bufsz, rep2:-CRep:-COEF, pvsz, pvbuf,
            cvsz, cvbuf, R:-prime):

       rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
       rep:-DGS:=dgsr:
       rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pvbuf, cvbuf, R:-VarList):
       rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
       rep:-CRep:-COEF:=rbuf:
       rep:-CRep:-PDGSV:=0:
       rep:-CRep:-COEFV:=0:
       rep:-CRepCase:=1:
    end if:

    if case=2 then
        ConnectorModule:-MTFTFFTC(n, shiftArrR(rep1:-DGS,n,1), (rep1:-CRep:-PDGSV)[1],
            rep1:-CRep:-PDGSV, (rep1:-CRep:-COEFV)[1], rep1:-CRep:-COEFV, shiftArrR(rep2:-DGS,n,1),
            (rep2:-CRep:-PDGSV)[1], rep2:-CRep:-PDGSV, (rep2:-CRep:-COEFV)[1],
            rep2:-CRep:-COEFV, shiftArrR(dgsr,n,1), pvsz, pvbuf, cvsz, cvbuf, R:-prime):

       rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
       rep:-DGS:=dgsr:
       rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pvbuf, cvbuf, R:-VarList):
       rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
       rep:-CRep:-COEF:=0:
       rep:-CRep:-PDGSV:=pvbuf:
       rep:-CRep:-COEFV:=cvbuf:
       rep:-CRepCase:=2:
    end if:
    return rep:
end proc;

#------------------------------------------------------------------------------
# Function:
#    {TFTFFTMulUNI}
# Briefly:
#    { Univariate polynomial multiplication based FFT or TFT. }
# Calling sequence:
#    { TFTFFTMulUNI(INrep1, INrep2, p) }
# Input:
#    { INrep1 : a univariate modpn polynomial }
#    { INrep2 : a univariate modpn polynomial }
#    { p : prime number }
# Output:
#    { The product of `INrep1` and 'INrep2` }
#------------------------------------------------------------------------------
TFTFFTMulUNI := proc (INrep1, INrep2, var, p)
   local dr, rep, rep1, rep2, res;
   rep1:=CRepAssertUNI(INrep1):
   rep2:=CRepAssertUNI(INrep2):
   dr:=rep1:-CRep:-DEG + rep2:-CRep:-DEG:
   if(dr>MEMORY_BOUND)  then error ERR_MEMORY: end if:
   res:=Array(1..dr+1, 'datatype' = 'integer[4]'):
   ConnectorModule:-TFTFFTUNI(dr, res, rep1:-CRep:-DEG, rep1:-CRep:-COEFV, rep2:-CRep:-DEG, rep2:-CRep:-COEFV, p):
   rep:=Record('RecdenPoly', 'CRep'):
   rep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, res, var):
   rep:-CRep:=Record('DEG', 'COEFV'):
   rep:-CRep:-DEG:=dr:
   rep:-CRep:-COEFV:=res:
   return rep:
end proc;

#------------------------------------------------------------------------------
# Function:
#    {removeLeadingZeros}
# Briefly:
#    {Remove the leading zeros input univariate polynomial}
# Calling sequence:
#    { removeLeadingZeros(crep)}
# Input:
#    { crep : a record representing a univariate modpn polynomial }
# Output:
#    { Same polynomial with leading degrees removed }
#    { If has leading zero, create a new vector, otherwise, use the origianl one.}
#------------------------------------------------------------------------------
removeLeadingZeros := proc(crep)
   local d, i, newCo, newcrep, newd;
   d:=crep:-DEG:
   for i from d+1 by -1 to 1 do
      newd:=i-1:
      if((crep:-COEFV)[i] <> 0) then
        break:
      end if:
   end do:
   if d = newd then
      newcrep:=crep:
   else
      if(newd>MEMORY_BOUND)  then error ERR_MEMORY: end if:
      newCo:=Array(1..newd+1, 'datatype' = 'integer[4]'):
      for i from 1 to newd+1 do
         newCo[i]:=(crep:-COEFV)[i]:
      end do:
      newcrep:=Record('DEG', 'COEFV'):
      newcrep:-DEG:=newd:
      newcrep:-COEFV:=newCo:
   end if:
   return newcrep:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {copyCRec}
# Briefly:
#   {Make a copy of the input C representation.}
# Calling sequence:
#   {copyCRec(crec)}
# Input:
#   {crec: C representation}
# Output:
#   {A copy of 'crec'.}
#------------------------------------------------------------------------------
copyCRec := proc(crec)
  local cpcrec, i, newV;
  cpcrec:=Record('DEG', 'COEFV'):
  cpcrec:-DEG:=crec:-DEG:
  if(cpcrec:-DEG > MEMORY_BOUND)  then error ERR_MEMORY: end if:
  newV:=Array(1..cpcrec:-DEG + 1, 'datatype' = 'integer[4]'):
  for i from 1 to cpcrec:-DEG + 1 do
     newV[i]:=(crec:-COEFV)[i]:
  end do:
  cpcrec:-COEFV:=newV:
  return cpcrec:
end proc:


# inner function.
copyRec := proc (rec)
  local cprec;
  cprec:=Record('RecdenPoly', 'CRep'):
  cprec:-RecdenPoly:=rec:-RecdenPoly:
  cprec:-CRep:=copyCRec(rec:-CRep):
  return cprec:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {createZeroRecUni}
# Briefly:
#   {Create a zero modpn polynomial}
# Calling sequence:
#   {createZeroRecUni()}
# Input:
#   { None }
# Output:
#   { The representation of the zero univariate modpn polynomial. }
#------------------------------------------------------------------------------
createZeroRecUni := proc()
  local cprec;
  cprec:=Record('RecdenPoly', 'CRep'):
  cprec:-CRep:=Record('DEG', 'COEFV'):
  cprec:-CRep:-DEG:=0:
  cprec:-CRep:-COEFV:=Array(1..1, 'datatype' = 'integer[4]'):
  cprec:-RecdenPoly:=PolynomialConvertOutUni(cprec:-CRep):
  return cprec:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {FastMonicDivUNI}
# Briefly:
#   {Monic fast univariate division}
# Calling sequence:
#   {FastMonicDivUNI(AINrep, BINrep, var, p)}
# Input:
#   { AINrep : univariate modpn polynomial }
#   { BINrep : univariate modpn polynomial }
#   { p : prime number }
# Output:
#   { A quotient-remainder record for the fast monic division of
#     `AINrep` by `BINrep ` . These are univariate modpn polynomials.}
#------------------------------------------------------------------------------
FastMonicDivUNI := proc (AINrep, BINrep, var, p)
   local Arep, Brep, dA, dB, qcrep, qd, qr, qrep, qv, rcrep, rd, rrep, rv, res;

   Arep:=CRepAssertUNI(AINrep):
   Brep:=CRepAssertUNI(BINrep):
   qr:=Record('Quo', 'Rem'):
   dA:=nops((op(Arep:-RecdenPoly))[2]):
   dB:=nops((op(Brep:-RecdenPoly))[2]):
   if dA < dB then
      qr:-Rem := copyRec(Arep):
      qr:-Quo:=createZeroRecUni():
      return qr:
   end if:
   rd:=Arep:-CRep:-DEG:
   if(rd > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   rv:=Array(1..rd+1, 'datatype' = 'integer[4]'):
   qd:=Arep:-CRep:-DEG - Brep:-CRep:-DEG:
   if(qd > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   qv:=Array(1..qd+1, 'datatype' = 'integer[4]'):
   ConnectorModule:-FASTDIVUNI(rd, rv, qd, qv, Arep:-CRep:-DEG, Arep:-CRep:-COEFV, Brep:-CRep:-DEG, Brep:-CRep:-COEFV, p):
   rcrep:=Record('DEG', 'COEFV'):
   rcrep:-DEG:=rd: rcrep:-COEFV:=rv:
   rcrep:=removeLeadingZeros(rcrep):
   qcrep:=Record('DEG', 'COEFV'):
   qcrep:-DEG:=qd: qcrep:-COEFV:=qv:
   qcrep:=removeLeadingZeros(qcrep):
   rrep:=Record('RecdenPoly', 'CRep'):
   rrep:-CRep:=rcrep:
   rrep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, rrep:-CRep:-COEFV, var):
   qrep:=Record('RecdenPoly', 'CRep'):
   qrep:-CRep:=qcrep:
   qrep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, qrep:-CRep:-COEFV, var):
   qr:-Quo:=qrep:
   qr:-Rem:=rrep:
   return qr:
end proc:

#------------------------------------------------------------------------------
# Function:
#    {PlainMonicDivUNI}
# Briefly:
#    {Classical univariate monic division.}
# Calling sequence:
#    { PlainMonicDivUNI(AINrep, BINrep, var, p)}
# Input:
#    { AINrep : univariate modpn polynomial. }
#    { BINrep : univariate modpn polynomial. }
#    { var : a variable. }
#    { p : prime number. }
# Output:
#    { A quotient-remainder record for the plain monic division of
#      `AINrep` by `BINrep `. These are univariate modpn polynomials. }
#------------------------------------------------------------------------------
PlainMonicDivUNI := proc (AINrep, BINrep, var, p)
   local Arep, Brep, dA, dB,qcrep, qd, qr, qrep, qv, rcrep, rd, rrep, rv,res;

   Arep:=CRepAssertUNI(AINrep):
   Brep:=CRepAssertUNI(BINrep):

   qr:=Record('Quo', 'Rem'):
   dA:=nops((op(Arep:-RecdenPoly))[2]):
   dB:=nops((op(Brep:-RecdenPoly))[2]):


   if dA < dB then
      qr:-Rem := copyRec(Arep):
      qr:-Quo:=createZeroRecUni():
      return qr:
   end if:

   rd:=Arep:-CRep:-DEG:
   if(rd > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   rv:=Array(1..rd+1, 'datatype' = 'integer[4]'):
   qd:=Arep:-CRep:-DEG - Brep:-CRep:-DEG:
   if(qd > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   qv:=Array(1..qd+1, 'datatype' = 'integer[4]'):
   ConnectorModule:-PLAINDIVUNI(rd, rv, qd, qv, Arep:-CRep:-DEG, Arep:-CRep:-COEFV, Brep:-CRep:-DEG, Brep:-CRep:-COEFV, p):
   rcrep:=Record('DEG', 'COEFV'):
   rcrep:-DEG:=rd: rcrep:-COEFV:=rv:
   rcrep:=removeLeadingZeros(rcrep):
   qcrep:=Record('DEG', 'COEFV'):
   qcrep:-DEG:=qd: qcrep:-COEFV:=qv:
   qcrep:=removeLeadingZeros(qcrep):
   rrep:=Record('RecdenPoly', 'CRep'):
   rrep:-CRep:=rcrep:
   rrep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, rrep:- CRep:-COEFV, var):
   qrep:=Record('RecdenPoly', 'CRep'):
   qrep:-CRep:=qcrep:
   qrep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, qrep:-CRep:-COEFV, var):
   qr:-Quo:=qrep:
   qr:-Rem:=rrep:
   return qr:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {LeadingCoefUNI}
# Briefly:
#   {Returns the leading coefficient of 'rpoly'.}
# Calling sequence:
#   {LeadingCoefUNI(rpoly)}
# Input:
#   { rpoly : a univariate Recden polynomial }
# Output:
#   { The leading coefficient of 'rpoly'.}
#------------------------------------------------------------------------------
LeadingCoefUNI:=proc(rpoly)
   local rop;
   rop:=op(rpoly):
   return rop[2][nops(rop[2])]:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {DegreeUNI}
# Briefly:
#   {Returns the degree of the input Recden polynomial.}
# Calling sequence:
#   {DegreeUNI(rpoly)}
# Input:
#   {rpoly : a univariate Recden polynomial. }
# Output:
#   { The degree of the input univariate Recden polynomial).
#------------------------------------------------------------------------------
DegreeUNI:=proc(rpoly)
   local rop;
   rop:=op(rpoly):
   return nops(rop[2])-1:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {CoMulRPolyUNI}
# Briefly:
#   {Multiplying a input coefficient with a input univariate Recden polynomial.}
# Calling sequence:
#   {CoMulRPolyUNI(co,rpoly,var,p)}
# Input:
#   {co: A coefficient.}
#   {rpoly: a univariate Recden polynomial.}
#   {var: a variable.}
#   {p: a prime number.}
# Output:
#   {The product of 'co' and 'rpoly' mod 'p'.}
#-----------------------------------------------------------------------------
CoMulRPolyUNI:=proc(co,rpoly,var,p)
   local L, i, n, newL, newrop, rop, tmp;
   rop:=op(rpoly):
   L:=rop[2]:
   n:=nops(L):
   newL:=[seq(0, i = 1 .. n)]:
   for i from 1 to n do
       tmp:=co*L[i] mod p:
       newL:=subsop(i=tmp,newL);
   end do:
   newrop := [p, [var], []], newL;
   return POLYNOMIAL(newrop):
end proc:

#------------------------------------------------------------------------------
# Function:
#   {MonicizeUNI}
# Briefly:
#   {Make the input univariate Recden polynomial monic.}
# Calling sequence:
#   { MonicizeUNI(rpoly, var, p)}
# Input:
#   { rpoly: a Recden polynomial.}
#   { var: a variable.}
#   { p: a prime number.}
# Output:
#   { Multiply the inverse of the lead coefficient of 'rpoly' with 'rpoly',}
#   { and returns the product.}
#-----------------------------------------------------------------------------
MonicizeUNI:=proc(rpoly, var, p)
   local lc, lcinv, rop;
   rop:=op(rpoly):
   lc:=(rop[2])[nops(rop[2])]:
   lcinv:= 1/lc mod p:
   return CoMulRPolyUNI(lcinv, rpoly, var, p):
end proc:


#------------------------------------------------------------------------------
# Function:
#    {PlainPseudoRemainder}
# Briefly:
#    {Classical Psedudo division.}
# Calling sequence:
#    {PlainPseudoRemainder(AINrep, BINrep, var, p)}
# Input:
#    { AINrep : univariate modpn polynomial }
#    { BINrep : univariate modpn polynomial }
#    { var : the variable }
#    { p : prime number }
# Output:
#    { The plain pseudo-division of `AINrep` by `BINrep ` }
#------------------------------------------------------------------------------
PlainPseudoRemainder := proc (AINrep, BINrep, var, p)
   local BMONICRPOLY, BMONICrep, Blc, co, e, r, rec, srep, srpoly;
   e:=DegreeUNI(AINrep:-RecdenPoly) - DegreeUNI(BINrep:-RecdenPoly):
   if(e<0) then return AINrep; end if:
   BMONICrep:=Record('RecdenPoly', 'CRep'):
   BMONICRPOLY:=MonicizeUNI(BINrep:-RecdenPoly, var, p):
   BMONICrep:-RecdenPoly:=BMONICRPOLY:
   BMONICrep:-CRep:=0:
   rec:=PlainMonicDivUNI(AINrep, BMONICrep, var, p):
   r:=rec:-Rem:-RecdenPoly:
   Blc:=LeadingCoefUNI(BINrep:-RecdenPoly):
   e:=e+1:
   co:=(Blc^e) mod p:
   r:=CoMulRPolyUNI(co, r, var, p):
   srpoly:=CoMulRPolyUNI(co, r, var, p):
   srep:=Record('RecdenPoly', 'CRep'):
   srep:-RecdenPoly:=r:
   srep:-CRep:=0:
   return srep:
end proc:


PlainXGCDUNIlocal:=proc(AINrep, BINrep, var, p)
   local  Arep, Brep, GCDcrep,GCDrep, GCDvec, Ucrep, Urep, Uvec, Vcrep, Vrep, Vvec, dA, dB, dGCD, dUV, qr;
   qr:=Record('U', 'V', 'GCD'):
   dA:=nops((op(AINrep:-RecdenPoly))[2]):
   dB:=nops((op(BINrep:-RecdenPoly))[2]):
   dUV:=dA+dB:
   if((2*dUV) > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   Uvec:=Array(1..dUV+1, 'datatype' = 'integer[4]'):
   Vvec:=Array(1..dUV+1, 'datatype' = 'integer[4]'):
   dGCD:=min(dA, dB):
   if(dGCD > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   GCDvec:=Array(1..dGCD+1, 'datatype' = 'integer[4]'):
   Arep:=CRepAssertUNI(AINrep):
   Brep:=CRepAssertUNI(BINrep):
   ConnectorModule:-PLAINGCDUNI(dUV, Uvec, dUV, Vvec, dGCD, GCDvec, Arep:-CRep:-DEG, Arep:-CRep:-COEFV, Brep:-CRep:-DEG, Brep:-CRep:-COEFV, p):
   Ucrep:=Record('DEG', 'COEFV'):
   Ucrep:-DEG:=dUV:
   Ucrep:-COEFV:=Uvec:
   Ucrep:=removeLeadingZeros(Ucrep):
   Vcrep:=Record('DEG', 'COEFV'):
   Vcrep:-DEG:=dUV:
   Vcrep:-COEFV:=Vvec:
   Vcrep:=removeLeadingZeros(Vcrep):
   GCDcrep:=Record('DEG', 'COEFV'):
   GCDcrep:-DEG:=dGCD:
   GCDcrep:-COEFV:=GCDvec:
   GCDcrep:=removeLeadingZeros(GCDcrep):
   Urep:=Record('RecdenPoly', 'CRep'):
   Urep:-CRep:=Ucrep:
   Urep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p,Urep:- CRep:-COEFV,var):
   Vrep:=Record('RecdenPoly', 'CRep'):
   Vrep:-CRep:=Vcrep:
   Vrep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, Vrep:- CRep:-COEFV, var):
   GCDrep:=Record('RecdenPoly', 'CRep'):
   GCDrep:-CRep:=GCDcrep:
   GCDrep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, GCDrep:- CRep:-COEFV, var):
   qr:-U:=Urep:
   qr:-V:=Vrep:
   qr:-GCD:=GCDrep:
   return qr:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {PlainXGCDUNI}
# Briefly:
#   {Classical Extended Euclidean GCD.}
# Calling sequence:
#   {PlainXGCDUNI(AINrep, BINrep, var, p)}
# Input:
#   { AINrep : univariate modpn polynomial }
#   { BINrep : univariate modpn polynomial }
#   { p : prime number }
# Output:
#   { The plain GCD of `AINrep` and `BINrep ` }
#------------------------------------------------------------------------------
PlainXGCDUNI:=proc(AINrep, BINrep, var, p)
    local qr, dA, dB;
    dA:=nops((op(AINrep:-RecdenPoly))[2]):
    dB:=nops((op(BINrep:-RecdenPoly))[2]):
    if dB>dA then
       qr:=PlainXGCDUNIlocal(BINrep, AINrep, var, p):
    else
       qr:=PlainXGCDUNIlocal(AINrep, BINrep, var, p):
    end if:
    return qr:
end proc:

FastXGCDUNIlocal:=proc(AINrep, BINrep, p)
    local Arep, Brep, GCDcrep, GCDrep, GCDvec, Ucrep, Urep, Uvec, Vcrep, Vrep, Vvec, dA, dB, dGCD, dUV,  qr;
    qr:=Record('U', 'V', 'GCD'):
    dA:=nops((op(AINrep:-RecdenPoly))[2]):
    dB:=nops((op(BINrep:-RecdenPoly))[2]):
    dUV:=dA+dB:
    if(2*dUV > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Uvec:=Array(1..dUV+1, 'datatype' = 'integer[4]'):
    Vvec:=Array(1..dUV+1, 'datatype' = 'integer[4]'):
    dGCD:=max(dA, dB):
    if(dGCD > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    GCDvec:=Array(1..dGCD+1, 'datatype' = 'integer[4]'):
    Arep:=CRepAssertUNI(AINrep):
    Brep:=CRepAssertUNI(BINrep):
    ConnectorModule:-FASTGCDUNI(dUV, Uvec, dUV, Vvec, dGCD, GCDvec, Arep:-CRep:-DEG, Arep:-CRep:-COEFV, Brep:-CRep:-DEG, Brep:-CRep:-COEFV, p):
    Ucrep:=Record('DEG', 'COEFV'):
    Ucrep:-DEG:=dUV:
    Ucrep:-COEFV:=Uvec:
    Ucrep:=removeLeadingZeros(Ucrep):
    Vcrep:=Record('DEG', 'COEFV'):
    Vcrep:-DEG:=dUV:
    Vcrep:-COEFV:=Vvec:
    Vcrep:=removeLeadingZeros(Vcrep):
    GCDcrep:=Record('DEG', 'COEFV'):
    GCDcrep:-DEG:=dGCD:
    GCDcrep:-COEFV:=GCDvec:
    GCDcrep:=removeLeadingZeros(GCDcrep):
    Urep:=Record('RecdenPoly', 'CRep'):
    Urep:-CRep:=Ucrep:
    Urep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p,Urep:- CRep:-COEFV, 'var'):
    Vrep:=Record('RecdenPoly', 'CRep'):
    Vrep:-CRep:=Vcrep:
    Vrep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, Vrep:- CRep:-COEFV, 'var'):
    GCDrep:=Record('RecdenPoly', 'CRep'):
    GCDrep:-CRep:=GCDcrep:
    GCDrep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, GCDrep:- CRep:-COEFV,'var'):
    qr:-U:=Urep:
    qr:-V:=Vrep:
    qr:-GCD:=GCDrep:
    return qr:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {FastXGCDUNI}
# Briefly:
#   {Fast Extend Euclidean GCD.}
# Calling sequence:
#   { FastXGCDUNI(AINrep, BINrep, p)}
# Input:
#   { AINrep : univariate modpn polynomial }
#   { BINrep : univariate modpn polynomial }
#   { p : prime number }
# Output:
#   { The fast extended GCD of `AINrep` and `BINrep ` }
#------------------------------------------------------------------------------
FastXGCDUNI:=proc(AINrep, BINrep, p)
    local qr, dA, dB;
    dA:=nops((op(AINrep:-RecdenPoly))[2]):
    dB:=nops((op(BINrep:-RecdenPoly))[2]):
    if dB>dA then
       qr := FastXGCDUNIlocal(BINrep, AINrep, p):
    else
       qr := FastXGCDUNIlocal(AINrep, BINrep, p):
    end if:
    return qr:
end proc:

log2Ceil:=proc(n)
    return ceil(log[2](n)):
end proc:

div2Ceil:=proc(n)
    return ceil(n/2.0):
end proc:

#------------------------------------------------------------------------------
# Function:
#   {SubProductTreeCreate}
# Briefly:
#   {To create a sub-product tree for fast evaluation.}
# Calling sequence:
#   {SubProductTreeCreate(itemNo, items, p)}
# Input:
#   { itemNo : number of points
#   { items : array of integers, the coordinates of the points }
#   { p : prime number }
# Output:
#   { The sub-product for fast univariate evaluation and interpolation }
#------------------------------------------------------------------------------
SubProductTreeCreate:=proc(itemNo, items, p)
    local  Bases, NoNodes, SubProdTree, W, base2, data, h, i, itemSz, levels, tmp, totSZ;
    itemSz := 2;
    h:=log2Ceil(itemNo):
    levels:=h+1:
    totSZ:=itemNo*itemSz:
    W:=Array(1..levels+1, 'datatype' = 'integer[4]'):
    NoNodes:=Array(1..levels+1, 'datatype' = 'integer[4]'):
    Bases:=Array(1..levels+1, 'datatype' = 'integer[4]'):
    W[2]:=itemSz: NoNodes[2]:=itemNo:
    for i from 3 to levels+1 do
       W[i]:=W[i-1]*2-1:
       NoNodes[i]:=div2Ceil(NoNodes[i-1]):
       totSZ:=totSZ+W[i]*NoNodes[i]:
    end do:
    if(totSZ > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    data:=Array(1..totSZ, 'datatype' = 'integer[4]'):
    tmp:=itemNo*itemSz:
    base2:=totSZ-tmp:
    # copyVec_0_to_d(tmp-1, (tree->data)+base2, items)
    for i from 1 to tmp-1+1 do
         data[base2+i]:=items[i]:
    end do:
    ConnectorModule:-SUBPTREECRE(h, levels, W, NoNodes, Bases, totSZ, data, itemNo, itemSz, p):
    SubProdTree:=Record('Rh', 'RW', 'RNoNodes', 'Rdata', 'RBases'):
    SubProdTree:-Rh:=h:  ## Heigth of the tree
    SubProdTree:-RW:=W:  ## maximum size of a polynomial at a leaf
    SubProdTree:-RNoNodes:=NoNodes:## Number of nodes
    SubProdTree:-Rdata:=data: ## the coefficients of the polynomials
         ## at the nodes and leaves, starting from the leaves,
         ## level by level.
    SubProdTree:-RBases:=Bases:
         ## the first index (in Rdata) of each level of the tree
    return SubProdTree:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {FastEvaluation}
# Briefly:
#   {Fast univariate evalution.}
# Calling sequence:
#   {FastEvaluation(df, fPtr, itemNo, tree, p)}
# Input:
#   { df :  integer, the degree of a polynomial }
#   { fPtr : array of integers, the coefficient vector of this
# polynomial }
#   { tree : a modpn subproduct tree (for evaluation/interpolation) }
#   { itemNo : the number of points associated with `tree` }
#   { p : prime number, the characteristic of the field }
# Output:
#   { a record of type Record('no', 'values') where 'no' is the number}
#   { of points, that is, `itemNo` and 'value' si the image vector. }
#------------------------------------------------------------------------------
FastEvaluation :=proc(df, fPtr, itemNo, tree, p)
   local EvalPts, EvalPtsRec:
    if(itemNo > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    EvalPts:=Array(1..itemNo, 'datatype' = 'integer[4]'):
    ConnectorModule:-FastEval(itemNo, EvalPts, df, fPtr, tree:-Rh, tree:-RW, tree:-RNoNodes,tree:-RBases, tree:-Rdata, p):
    EvalPtsRec:=Record('no', 'values'):
    EvalPtsRec:-no:=itemNo:
    EvalPtsRec:-values:=EvalPts:
    return EvalPtsRec:
end proc:

#------------------------------------------------------------------------------
# Function:
#    {FastInterpolation}
# Briefly:
#    {Fast univariate interpolation.}
# Calling sequence:
#    {FastInterpolation(itemNo, EvaluatingPts, EvaluatedPts, tree, p)}
# Input:
#    { tree : a modpn subproduct tree (for evaluation/interpolation) }
#    { itemNo : the number of points associated with `tree` }
#    { p : prime number, the characteristic of the field }
#    { EvaluatingPts : the evaluating points defining the `tree` }
#    { EvaluatedPts : the images at these points of the polynomial to be
#    interpolated }
# Output:
#    { a record of type Record('DEG', 'COEFV') where 'DEG' and 'COEFV' }
#    { are the degree and the coefficient vector of the interpolated }
#    { polynomial. These are C objects. }
#------------------------------------------------------------------------------
FastInterpolation :=proc(itemNo, EvaluatingPts, EvaluatedPts, tree, p)
    local Coeffs, EvalPtsRec,  rcrep;
    if(itemNo > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Coeffs:=Array(1..itemNo, 'datatype' = 'integer[4]'):
    ConnectorModule:-FastInterp(itemNo, Coeffs, EvaluatingPts, EvaluatedPts, tree:-Rh,
            tree:-RW, tree:-RNoNodes, tree:-RBases, tree:-Rdata, p):
    #EvalPtsRec:=Record('no', 'values'):
    rcrep:=Record('DEG', 'COEFV'):
    rcrep:-DEG:=itemNo-1: rcrep:-COEFV:=Coeffs:
    rcrep:=removeLeadingZeros(rcrep):
    return rcrep:
end proc:

#------------------------------------------------------------------------------
# Function:
#    {FastInterpRatFunCons}
# Briefly:
#    {Fast interpolate then rational function reconstruction}
#    {from the interpolated univariate polynomial.}
# Calling sequence:
#    {FastInterpRatFunCons(itemNo, EvaluatingPts, EvaluatedPts, tree, var, p)}
# Input:
#    {itemNo: Number of points for the evalution.}
#    {EvaluatingPts: The points at which the evalutions are performed.}
#    {EvaluatedPts: The evalutated points (= images).}
#    {tree: the sub-product tree.}
#    {var: the variable for the univariate rational function.}
#    {p: the prime number.}
# Output:
#    {The rational function, as a list of two univariate modpn polynomials}
#------------------------------------------------------------------------------
FastInterpRatFunCons :=proc(itemNo, EvaluatingPts, EvaluatedPts, tree, var, p)
    local Coeffs, EvalPtsRec,  rcrep, mymsg, CoeffsNum, CoeffsDen, itemNop,
    rcrepnum, repnum, rcrepden, repden;

    if((2*itemNo) > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    CoeffsNum:=Array(1..itemNo, 'datatype' = 'integer[4]'):
    CoeffsDen:=Array(1..itemNo, 'datatype' = 'integer[4]'):
    itemNop:=Array(1..2, 'datatype' = 'integer[4]'):
    itemNop[1]:=itemNo:
    itemNop[2]:=p:
    mymsg:=ConnectorModuleNew1:-FastInterpRatFunCons(itemNop, CoeffsNum,
        CoeffsDen, EvaluatingPts, EvaluatedPts, tree:-Rh, tree:-RW, tree:-RNoNodes,
        tree:-RBases, tree:-Rdata):

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:

    #EvalPtsRec:=Record('no', 'values'):
    rcrepnum:=Record('DEG', 'COEFV'):
    rcrepnum:-DEG:=itemNo-1: rcrepnum:-COEFV:=CoeffsNum:
    rcrepnum:=removeLeadingZeros(rcrepnum):
    repnum:=Record('RecdenPoly', 'CRep'):
    repnum:-CRep:=rcrepnum:
    repnum:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, repnum:-CRep:-COEFV, var):

    rcrepden:=Record('DEG', 'COEFV'):
    rcrepden:-DEG:=itemNo-1: rcrepden:-COEFV:=CoeffsDen:
    rcrepden:=removeLeadingZeros(rcrepden):
    repden:=Record('RecdenPoly', 'CRep'):
    repden:-CRep:=rcrepden:
    repden:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(p, repden:-CRep:-COEFV, var):
    return [repnum, repden]:
end proc:

#------------------------------------------------------------------------------
# Function:
#   {CreatePointsTrees}
# Briefly:
#   {Create the points and product trees for multivariate evalution.}
# Calling sequence:
#   {CreatePointsTrees(N, M, ININrep1, ININrep2, p)  }
# Input:
#   { INrep1 : a multivariate modpn polynomial }
#   { INrep2 : a multivariate modpn polynomial }
#   { p : prime number, the characteristic of the field }
#   { N : the total number of variables appearing in `INrep1` and
#     `INrep2` }
#   { M : the number of variables to be evaluated }
# Output:
#   { a record holding the points and the sub-product trees needed to
#     evaluate `INrep1` and `INrep2` and interpolate their sub-resultant
#     chain. Moreoover, these evaluation points are chosen such that the
#     leading coefficients of `INrep1` and `INrep2` are not cancelled. }
#------------------------------------------------------------------------------

# bounds[2..N+1] keeps the data.
CreatePointsTrees:=proc(N, M, ININrep1, ININrep2, p)
    local mymsg, Basess, NMp, NoNodess, PHWD, PTSTREE, WNBsSz, Ws, bounds,
    dataSz, datas, datasSz, dims1, dims2, hs, hsSz, i, itemSz, p1Sz, p2Sz,
    ptss, ptssSz, rep1, rep2, thelevels, INrep1, INrep2, V;

    INrep1:=PartialDegsAssert(ININrep1):
    INrep2:=PartialDegsAssert(ININrep2):

    rep1:=CRepAssert(INrep1, 1):
    rep2:=CRepAssert(INrep2, 1):
    bounds:=Array(1..M+1, 'datatype' = 'integer[4]'):
    dims1:=Array(1..N+1, 'datatype' = 'integer[4]'):
    dims2:=Array(1..N+1, 'datatype' = 'integer[4]'):

    for i from M+2 to N+1 do
      dims1[i]:=(rep1:-DGS[i-1])+1:
      dims2[i]:=(rep2:-DGS)[i-1]+1:
    end do:

    ## variable list
    V := op(2, op(1, rep1[RecdenPoly]));
    for i from 2 to M+1 do
        ## bounds[i] := (rep1:-DGS)[i-1]*(rep2:-DGS[N]) + (rep2:-DGS)[i-1]*(rep1:-DGS)[N] +1:
        bounds[i] := MinimalDegreeBound(PolynomialConvertOut(rep1),
            PolynomialConvertOut(rep2), V[1], V[-i+1]) + 1;
        dims1[i] := bounds[i];
        dims2[i] := bounds[i];
    end do;

    ptssSz:=0:
    for i from 2 to M+1 do
        ptssSz:=ptssSz+bounds[i]:
    end do:
    if(ptssSz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    ptss:=Array(1..ptssSz, 'datatype' = 'integer[4]'):
    hsSz:=M:
    if(hsSz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    hs:=Array(1..hsSz, 'datatype' = 'integer[4]'):
    WNBsSz:=0:
    for i from 2 to M+1 do  WNBsSz:=2*2*M*(log2Ceil(bounds[i])+1+1+1): end do:
    if(3*WNBsSz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Ws:=Array(1..WNBsSz, 'datatype' = 'integer[4]'):
    NoNodess:=Array(1..WNBsSz, 'datatype' = 'integer[4]'):
    Basess:=Array(1..WNBsSz, 'datatype' = 'integer[4]'):
    dataSz:=0: datasSz:=0: itemSz:=2:
    for i from 2 to M+1 do
       thelevels:=log2Ceil(bounds[i])+1:
       dataSz:=bounds[i]*itemSz*thelevels:
       datasSz:=datasSz+dataSz:
    end do:
    if(3*datasSz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    datas:=Array(1..datasSz, 'datatype' = 'integer[4]'):
    p1Sz:=calSizDgs(rep1:-DGS,N):
    p2Sz:=calSizDgs(rep2:-DGS,N):

    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    PHWD:=Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1]:=ptssSz: PHWD[2]:=hsSz: PHWD[3]:=WNBsSz: PHWD[4]:=dataSz:

    mymsg:=ConnectorModuleNew1:-CrePtsPTrees (NMp, PHWD, bounds, dims1, dims2,
        ptss, hs, Ws, NoNodess, Basess, datas, rep1:-DGS, p1Sz, rep1:-CRep:-COEF,
        rep2:-DGS, p2Sz, rep2:-CRep:-COEF):

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110) then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:

    PTSTREE:=Record( 'PThsSz', 'PThs', 'PTWNBsSz', 'PTWs', 'PTNoNodess',
        'PTBasess', 'PTdatasSz', 'PTdatas', 'PTbounds', 'PTdims1', 'PTdims2',
        'PTptssSz', 'PTptss'):

    PTSTREE:-PThsSz:=hsSz:
    ## PThsSz is the size of PThs (an array)
    PTSTREE:-PThs:=hs:
    ## PThs the array of the heigths of each tree
    PTSTREE:-PTWNBsSz:=WNBsSz:
    ## PTWNBsSz is the size of -PTWs (an array)
    PTSTREE:-PTWs:=Ws:
    ## array of the sizes of the polynomial leaves in the
    ## trees. The order of these trees is from the greatest
    ## variable to be evaluated to the least one.
    PTSTREE:-PTNoNodess:=NoNodess:
    ## numbers of nodes for each tree
    PTSTREE:-PTBasess:=Basess:
    ## the starting indices for each tree
    PTSTREE:-PTdatasSz:=datasSz:
    ## the data sizes
    PTSTREE:-PTdatas:=datas:
    ## the data of the trees, consecutively, in one vector
    PTSTREE:-PTbounds:=bounds:
    ## bounds for interpolating the subresultants
    PTSTREE:-PTdims1:=dims1:
    ## coefficient sizes in a recursive vision (see code)
    PTSTREE:-PTdims2:=dims2:
    ## coefficient sizes in a recursive vision (see code)
    PTSTREE:-PTptssSz:=ptssSz:
    ## Array of size equal to the number of specialized variables
    PTSTREE:-PTptss:=ptss:
    ## the points at which each variable is specialized,
    ## one variable after another.
    return PTSTREE:
end proc:

## Similar to the previous function.
## The difference is that it is meant to compute
## (exact) division instead of subresultants
CreatePointsTreesDiv:=proc(N, M, ININrep1, ININrep2, p)
    local Basess, NMp, NoNodess, PHWD, PTSTREE, WNBsSz, Ws, bounds, dataSz,
    datas, datasSz, dims1, dims2, hs, hsSz, i, itemSz, p1Sz, p2Sz, ptss,
    ptssSz, rep1, rep2, thelevels, INrep1, INrep2;

    INrep1:=PartialDegsAssert(ININrep1):
    INrep2:=PartialDegsAssert(ININrep2):
    rep1:=CRepAssert(INrep1, 1):
    rep2:=CRepAssert(INrep2, 1):
    bounds:=Array(1..M+1, 'datatype' = 'integer[4]'):
    dims1:=Array(1..N+1, 'datatype' = 'integer[4]'):
    dims2:=Array(1..N+1, 'datatype' = 'integer[4]'):

    for i from M+2 to N+1 do
      dims1[i]:=(rep1:-DGS[i-1])+1:
      dims2[i]:=(rep2:-DGS)[i-1]+1:
    end do:
    for i from 2 to M+1 do

       bounds[i]:=(rep1:-DGS)[i-1] + ((rep1:-DGS)[N] - (rep2:-DGS)[N]) * (rep2:-DGS)[i-1]:
       dims1[i]:=bounds[i]:
       dims2[i]:=bounds[i]:
    end do:
    ptssSz:=0:
    for i from 2 to M+1 do
        ptssSz:=ptssSz+bounds[i]:
    end do:
    if(ptssSz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    ptss:=Array(1..ptssSz, 'datatype' = 'integer[4]'):
    hsSz:=M:
    if(hsSz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    hs:=Array(1..hsSz, 'datatype' = 'integer[4]'):
    WNBsSz:=0:
    for i from 2 to M+1 do  WNBsSz:=M*(log2Ceil(bounds[i])+1+1+1): end do:
    if(3*WNBsSz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Ws:=Array(1..WNBsSz, 'datatype' = 'integer[4]'):
    NoNodess:=Array(1..WNBsSz, 'datatype' = 'integer[4]'):
    Basess:=Array(1..WNBsSz, 'datatype' = 'integer[4]'):
    dataSz:=0: datasSz:=0: itemSz:=2:
    for i from 2 to M+1 do
       thelevels:=log2Ceil(bounds[i])+1:
       dataSz:=bounds[i]*itemSz*thelevels:
       datasSz:=datasSz+dataSz:
    end do:
    if(datasSz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    datas:=Array(1..datasSz, 'datatype' = 'integer[4]'):
    p1Sz:=calSizDgs(rep1:-DGS,N):
    p2Sz:=calSizDgs(rep2:-DGS,N):

    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    PHWD:=Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1]:=ptssSz: PHWD[2]:=hsSz: PHWD[3]:=WNBsSz: PHWD[4]:=dataSz:

    ConnectorModuleNew1:-CrePtsPTrees (NMp, PHWD, bounds, dims1, dims2, ptss,
        hs, Ws, NoNodess, Basess, datas, rep1:-DGS, p1Sz, rep1:-CRep:-COEF,
        rep2:-DGS, p2Sz, rep2:-CRep:-COEF):

    PTSTREE:=Record( 'PThsSz', 'PThs', 'PTWNBsSz', 'PTWs', 'PTNoNodess', 'PTBasess',
        'PTdatasSz', 'PTdatas', 'PTbounds', 'PTdims1', 'PTdims2', 'PTptssSz', 'PTptss'):

    PTSTREE:-PThsSz:=hsSz:
    PTSTREE:-PThs:=hs:
    PTSTREE:-PTWNBsSz:=WNBsSz:
    PTSTREE:-PTWs:=Ws:
    PTSTREE:-PTNoNodess:=NoNodess:
    PTSTREE:-PTBasess:=Basess:
    PTSTREE:-PTdatasSz:=datasSz:
    PTSTREE:-PTdatas:=datas:
    PTSTREE:-PTbounds:=bounds:
    PTSTREE:-PTdims1:=dims1:
    PTSTREE:-PTdims2:=dims2:
    PTSTREE:-PTptssSz:=ptssSz:
    PTSTREE:-PTptss:=ptss:
    return PTSTREE:
end proc:

#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# { INrep :  a multivariate modpn polynomial }
# { p : prime number, the characteristic of the field }
# { N : the total number of variables appearing in `INrep` }
# { M : the number of variables to be evaluated }
# { PSTREE : the data-structure holding the points and the sub-product
#  trees needed to evaluate `INrep` }
# Output:
# { rep:-DGS : partial degrees of the input polynomial. Ask Xin for details. }
# { PTSTREE:-PTdims1 :
# { E : an evaluation representation of the `INrep` where the `M`
# lowest variables are evaluated }
# { Esz : an integer, the total number of slots of `E` }
# {  PTSTREE:-PTdims1 : a one-dimensional array of size `N+1` where
# the first slot is not used and the others give the diemnsions of `E`
# from the smallest to the largest variable in `INrep`
# E.g.  (PTSTREE:-PTdims1)[2] is the first  dimension  of 'E' .}
#------------------------------------------------------------------------------
FastEvaluationM1:=proc(N, M, ININrep, PTSTREE, p)
    local mymsg, Ecube, Edat, Esz, NMp, PHWD, i, pSz, rep, INrep;

    INrep:=PartialDegsAssert(ININrep):
    rep:=CRepAssert(INrep, 1):
    pSz:=calSizDgs(rep:-DGS,N):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    PHWD:=Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1]:=PTSTREE:-PTptssSz: PHWD[2]:=PTSTREE:-PThsSz: PHWD[3]:=PTSTREE:-PTWNBsSz: PHWD[4]:=PTSTREE:-PTdatasSz:
    Esz:=1:
    for i from 2 to N+1 do
        Esz:=Esz*(PTSTREE:-PTdims1)[i]:
    end do:
    if(Esz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Edat:=Array(1..Esz, 'datatype' = 'integer[4]'):
    mymsg:=ConnectorModuleNew1:-FastEvalM (NMp, PHWD, PTSTREE:-PTbounds,
        PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs,  PTSTREE:-PTNoNodess,
        PTSTREE:-PTBasess, PTSTREE:-PTdatas, PTSTREE:-PTdims1, Esz,
        Edat, rep:-DGS, pSz, rep:-CRep:-COEF):

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    Ecube:=Record( 'Edims', 'Esize', 'Edata'):
    Ecube:-Edims:=PTSTREE:-PTdims1:
    Ecube:-Esize:=Esz:
    Ecube:-Edata:=Edat:
    return Ecube:
end proc:

#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# {
# {
# Output:
# {
# {
#------------------------------------------------------------------------------
FastEvaluationM2:=proc(N, M, ININrep, PTSTREE, p)
    local  Ecube, Edat, Esz, NMp, PHWD, i, pSz, rep, INrep, mymsg;
    INrep:=PartialDegsAssert(ININrep):
    rep:=CRepAssert(INrep, 1):
    pSz:=calSizDgs(rep:-DGS, N):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    PHWD:=Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1]:=PTSTREE:-PTptssSz: PHWD[2]:=PTSTREE:-PThsSz:
    PHWD[3]:=PTSTREE:-PTWNBsSz: PHWD[4]:=PTSTREE:-PTdatasSz:
    Esz:=1:
    for i from 2 to N+1 do
        Esz:=Esz*(PTSTREE:-PTdims2)[i]:
    end do:
    if(Esz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Edat:=Array(1..Esz, 'datatype' = 'integer[4]'):
    mymsg:=ConnectorModuleNew1:-FastEvalM (NMp, PHWD, PTSTREE:-PTbounds,
        PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs,  PTSTREE:-PTNoNodess,
        PTSTREE:-PTBasess, PTSTREE:-PTdatas, PTSTREE:-PTdims2, Esz, Edat,
        rep:-DGS, pSz, rep:-CRep:-COEF):
    
    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110) then error  ERR_PRIME: end if:
     if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    Ecube:=Record( 'Edims', 'Esize', 'Edata'):
    Ecube:-Edims:=PTSTREE:-PTdims2:
    Ecube:-Esize:=Esz:
    Ecube:-Edata:=Edat:
    return Ecube:
end proc:

#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables appearing in `INrep` }
# { M : the number of variables to be evaluated }
# { w : the minimum degree of the input polynomials w.r.t. their
# (common) main variable. }
# { bounds : a one-dimensional array containing (plus-one) bounds for
# the resultant of the input polynomials w.r.t. their other
# variables, that is, X_1, ..., X_M . Note the first slot is not
# used. }
# { Edims1, E1Sz, E1: the evaluation representation of the first
# polynomial. }
# { Edims2, E2Sz, E2: the evaluation representation of the first
# polynomial. }
# Output:
# { S : the sub-resultant chains of the input polynomials evaluated at
# each point of their evaluation representation. }
# { Ssz : the total number of slots in 'S' }
# { Sdims : the dimensions of S, that is, 'bounds' followed by '[w,w]'.}
#------------------------------------------------------------------------------
GetResultantChains:=proc(N, M,w, bounds, Ecube1, Ecube2, p)
    local mymsg, Scube, Sdat, Ssz, i, ssdims;
    Ssz:=1:
    for i from 2 to M+1 do    Ssz:=Ssz*bounds[i]:  end do:
    Ssz:=Ssz*w*w:
    if(Ssz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    userinfo(8, MODPN, "Ssz=", Ssz);
    Sdat:=Array(1..Ssz, 'datatype' = 'integer[4]'):
    mymsg:=ConnectorModuleNew1:-GetResChains(N, w, Ssz, Sdat,  Ecube1:-Edims,
        Ecube1:-Esize, Ecube1:-Edata, Ecube2:-Edims, Ecube2:-Esize, Ecube2:-Edata, p):

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    ssdims:=Array(1..N+2, 'datatype'='integer[4]'):
    for i from 2 to N do ssdims[i]:=bounds[i]: end do:
    ssdims[N+1]:=w: ssdims[N+2]:=w:
    Scube:=Record( 'Sdims', 'Ssize', 'Sdata', 'SLc', 'SPoly'):
    Scube:-Sdims:=ssdims:
      ##
    Scube:-Ssize:=Ssz:
    Scube:-Sdata:=Sdat:
    Scube:-SLc:=Array(1..w):
      ## buffer space for the initials of the subresultants
    Scube:-SPoly:=Array(1..w):
      ## buffer space for the subresultants
    return Scube:
end proc:

GetResultantChainsDft:=proc(N, M,w, bounds, Ecube1, Ecube2, p)
    local mymsg, Scube, Sdat, Ssz, i, ssdims;
    Ssz:=1:
    for i from 2 to M+1 do    Ssz:=Ssz*bounds[i]:  end do:
    Ssz:=Ssz*w*w:
    if(Ssz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Sdat:=Array(1..Ssz, 'datatype' = 'integer[4]'):
    userinfo(8, MODPN, "Ssz=", Ssz);
    mymsg:=ConnectorModuleNew1:-GetResChains(N, w, Ssz, Sdat, 
        Ecube1:-Edims, Ecube1:-Esize, Ecube1:-Edata, Ecube2:-Edims,
        Ecube2:-Esize, Ecube2:-Edata, p):

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    ssdims:=Array(1..N+2, 'datatype'='integer[4]'):
    for i from 2 to N do ssdims[i]:=bounds[i]: end do:
    ssdims[N+1]:=w: ssdims[N+2]:=w:
    Scube:=Record( 'Ses', 'Sdims', 'Ssize', 'Sdata', 'SLc', 'SPoly'):
    Scube:-Ses:=Ecube1:-Ees;
    Scube:-Sdims:=ssdims:
    Scube:-Ssize:=Ssz:
    Scube:-Sdata:=Sdat:
    Scube:-SLc:=Array(1..w):
    Scube:-SPoly:=Array(1..w):
    return Scube:
end proc:




# opt = 0  => Eclidean Quo.
# opt = 1  => Psedudo Quo.
# Subproduct-tree based approach
# dd is the degree of theqoutient
GetQuotientImage:=proc(N, M, dd, bounds, Ecube1, Ecube2, p, opt)
    local Qcube, Qdat, Qsz, i, ssdims;
    Qsz:=1:
    for i from 2 to M+1 do    Qsz:=Qsz*bounds[i]:  end do:
    Qsz:=Qsz*(dd+1):
    if(Qsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Qdat:=Array(1..Qsz, 'datatype' = 'integer[4]'):
    ConnectorModuleNew1:-GetQuotientImage(N, dd, Qsz, Qdat, Ecube1:-Edims,
        Ecube1:-Esize, Ecube1:-Edata, Ecube2:-Edims, Ecube2:-Esize,
        Ecube2:-Edata, p, opt):

    ssdims:=Array(1..N+2, 'datatype'='integer[4]'):
    for i from 2 to N do ssdims[i]:=bounds[i]: end do:
    ssdims[N+1]:=dd+1:
    Qcube:=Record( 'Qdims', 'Qsize', 'Qdata'):
    Qcube:-Qdims:=ssdims:
    Qcube:-Qsize:=Qsz:
    Qcube:-Qdata:=Qdat:
    return Qcube:
end proc:

DftGetQuotientImage:=proc(N, M, dd, bounds, Ecube1, Ecube2, p, opt)
    local Qcube, Qdat, Qsz, i, ssdims;
    Qsz:=1:
    for i from 2 to M+1 do    Qsz:=Qsz*bounds[i]:  end do:
    Qsz:=Qsz*(dd+1):
    if(Qsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Qdat:=Array(1..Qsz, 'datatype' = 'integer[4]'):
    ConnectorModuleNew1:-GetQuotientImage(N, dd, Qsz, Qdat,
    Ecube1:-Edims, Ecube1:-Esize, Ecube1:-Edata,
    Ecube2:-Edims, Ecube2:-Esize, Ecube2:-Edata, p, opt):
    ssdims:=Array(1..N+2, 'datatype'='integer[4]'):
    for i from 2 to N do ssdims[i]:=bounds[i]: end do:
    ssdims[N+1]:=dd+1:
    Qcube:=Record( 'Qes', 'Qdims', 'Qsize', 'Qdata'):
    Qcube:-Qes:=Ecube1:-Ees;
    Qcube:-Qdims:=ssdims:
    Qcube:-Qsize:=Qsz:
    Qcube:-Qdata:=Qdat:
    return Qcube:
end proc:
#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables appearing }
# { M : the number of variables to be evaluated }
# { R : a modpn polynomial ring of characteristic `p` and with
# variable list }
# { PTSTREE : sub-product tree data-structure. }
# { ESTIDGS : esimate partial degrees for the output polynomial }
# { Edims, Esz, E : output of the evaluation }
# Output:
# { a modpn multivariate polynomial in recden format }
#------------------------------------------------------------------------------
FastInterpolationM:=proc(R, N, M, PTSTREE, ESTIDGS, Ecube, p)
   local mymsg, E, Esz, NMp, PHWD,cVsz, coefVec, dVsz, edims, pdegVec, rep;
   edims:=Ecube:-Edims:
   Esz:=Ecube:-Esize:
   E:=Ecube:-Edata:
   dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
   if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
   cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
   if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
   NMp:=Array(1..3, 'datatype' = 'integer[4]'):
   NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
   PHWD:=Array(1..4, 'datatype' = 'integer[4]'):
   PHWD[1]:=PTSTREE:-PTptssSz: PHWD[2]:=PTSTREE:-PThsSz: PHWD[3]:=PTSTREE:-PTWNBsSz: PHWD[4]:=PTSTREE:-PTdatasSz:

   mymsg:=ConnectorModuleNew1:-FastInterpM(NMp, PHWD, PTSTREE:-PTbounds,
    PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs,  PTSTREE:-PTNoNodess, 
    PTSTREE:-PTBasess, PTSTREE:-PTdatas, edims, Esz, E, dVsz, pdegVec, cVsz, coefVec):

  if mymsg=-10  then error  ERR_INTERRUPT: end if:
  if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
  if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
  rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
  rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
  #rep:-DGS:=(RecdenConnector:-PartialDegs(RPoly))[2]:
  rep:-DGS:=0:
  rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
  rep:-CRep:-COEF:=0:
  rep:-CRep:-PDGSV:=pdegVec:
  rep:-CRep:-COEFV:=coefVec:
  rep:-CRepCase:=2:
  return rep:
end proc:

#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# {
# {
# Output:
# {
# {
#------------------------------------------------------------------------------
## To interpolate the ith subresultant from the Scube
FastInterpolatePolynomial:=proc(R, N, M, ith, PTSTREE, w, Scube, p)
    local ESTIDGS, NMp, PHWD, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep, slicedims, slicesz;
    if(((Scube:-SPoly)[ith+1]) <> 0) then
         return (Scube:-SPoly)[ith+1]:
    end if:

    slicedims:=Scube:-Sdims:
    Ssz:=Scube:-Ssize:
    S:=Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := PTSTREE:-PTbounds[i] -1 end do:
    ESTIDGS[N+1] := w-1:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    PHWD:=Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1]:=PTSTREE:-PTptssSz: PHWD[2]:=PTSTREE:-PThsSz: PHWD[3]:=PTSTREE:-PTWNBsSz: PHWD[4]:=PTSTREE:-PTdatasSz:
    slicesz:=iquo(Ssz,w):
    ConnectorModuleNew1:-FastInterpithM(NMp, PHWD, ith, PTSTREE:-PTbounds,
        PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs,  PTSTREE:-PTNoNodess,
        PTSTREE:-PTBasess, PTSTREE:-PTdatas, w, slicesz, slicedims, Ssz,
        S, dVsz, pdegVec, cVsz, coefVec):

    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):

    #rep:-RecdenPoly := Algebraic:-RecursiveDensePolynomials:-pprpoly(rep:-RecdenPoly, (R:-VarList)[1]);
    #rep:-DGS:=(RecdenConnector:-PartialDegs(rep:-RecdenPoly))[2]:
    rep:-DGS:=0:
    #rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
    #rep:-CRep:-COEF:=0:
    #rep:-CRep:-PDGSV:=pdegVec:
    #rep:-CRep:-COEFV:=coefVec:
    rep:-CRep:=0:
    rep:-CRepCase:=-1:

    if(nargs=9) then
      rep:=ReduceCoeffs(R, rep, args[9]):
    end if:
    (Scube:-SPoly)[ith+1]:=rep:
    return rep:
        end proc:

## N: the number of variables from the least one to the main one.
## M: the number of variables has been evaluated.
## PTSTREE: The points and subproduct-tree used conduct the evaluation.
## bb: the degree difference in main variable between two polynomials/
FastInterpolatePolynomialDiv:=proc(R, N, M,  PTSTREE, bb, Qcube, p)
       local ESTIDGS, NMp, PHWD, Q, Qsz, cVsz, coefVec, dVsz, i, pdegVec, rep, dims, slicesz, mymsg;

       dims:=Qcube:-Qdims:

       Qsz:=Qcube:-Qsize:
       Q:=Qcube:-Qdata:
       ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
       for i from 2 to M+1 do ESTIDGS[i] := PTSTREE:-PTbounds[i]  end do:
       ESTIDGS[N+1] := bb:
       dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
       if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
       pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
       cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
       if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
       coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
       NMp:=Array(1..3, 'datatype' = 'integer[4]'):
       NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
       PHWD:=Array(1..4, 'datatype' = 'integer[4]'):
       PHWD[1]:=PTSTREE:-PTptssSz: PHWD[2]:=PTSTREE:-PThsSz:
       PHWD[3]:=PTSTREE:-PTWNBsSz: PHWD[4]:=PTSTREE:-PTdatasSz:
       mymsg:=ConnectorModuleNew1:-FastInterpM(NMp, PHWD, PTSTREE:-PTbounds,
        PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs,  PTSTREE:-PTNoNodess,
        PTSTREE:-PTBasess, PTSTREE:-PTdatas, dims, Qsz, Q, dVsz, pdegVec, cVsz, coefVec):

      if mymsg=-10  then error  ERR_INTERRUPT: end if:
      if (mymsg=-100 or mymsg=-110) then error  ERR_PRIME: end if:
       if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
       rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
       rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
       rep:-DGS:=0:
       rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
       rep:-CRep:-COEF:=0:
       ## the C-Cube data encoding is not provided.
       rep:-CRep:-PDGSV:=pdegVec:
       rep:-CRep:-COEFV:=coefVec:
       rep:-CRepCase:=2:
       ## the C-2Vector data encoding is return from C level and saved.
       return rep:
end proc:

## The same procedure as the above one, except using DFT.
FastInterpolatePolynomialDivDft:=proc(R, N, M, ROOTS, dd, Qcube, p)
       local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep, slicedims, slicees, slicesz, Qes, Qdims, Qsz, Q;
       Qes:=Qcube:-Qes:
       Qdims:=Qcube:-Qdims:
       Qsz:=Qcube:-Qsize:
       Q:=Qcube:-Qdata:
       ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
       for i from 2 to M+1 do ESTIDGS[i] := Qdims[i] -1 end do:
       ESTIDGS[N+1] := dd:
       dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
       if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
       pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
       cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
       if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
       coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
       NMp:=Array(1..3, 'datatype' = 'integer[4]'):
       NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
       ConnectorModuleNew1:-DftInterpM(NMp, Qes, Qdims, Qsz, Q, dVsz, pdegVec, cVsz, coefVec, ROOTS):
       rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
       rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
       rep:-DGS:=0:
       rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
       rep:-CRep:-COEF:=0:
       rep:-CRep:-PDGSV:=pdegVec:
       rep:-CRep:-COEFV:=coefVec:
       rep:-CRepCase:=2:
       return rep:
end proc:
#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# {
# {
# Output:
# {
#------------------------------------------------------------------------------
## N: the number of variables from the least one to the main one.
## M: the number of variables has been evaluated.
## ith: ith row in the subresultant chain.
## dth: dth column in the subresultant chain.
## PTSTREE: The points and subproduct-tree used conduct the evaluation.
## w = degree(poly2)-1.
## Interpolate a coefficient from the subresultant chain.

FastInterpolateCoefficient:=proc(R, N, M, ith, dth, PTSTREE, w, Scube, p)
    local ESTIDGS, NMp, PHWD, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep,
    slicedims, slicesz, tmp, ww;

    slicedims:=Scube:-Sdims:
    Ssz:=Scube:-Ssize:
    S:=Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := PTSTREE:-PTbounds[i] -1 end do;
    ESTIDGS[N+1] := 0:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:

    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    PHWD:=Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1]:=PTSTREE:-PTptssSz: PHWD[2]:=PTSTREE:-PThsSz:
    PHWD[3]:=PTSTREE:-PTWNBsSz: PHWD[4]:=PTSTREE:-PTdatasSz:
    ww:=w*w;
    slicesz:=iquo(Ssz,ww):
    tmp:=slicedims[N+1]:
    slicedims[N+1]:=1:

    ConnectorModuleNew1:-FastInterpithdthM(NMp, PHWD, ith, dth, PTSTREE:-PTbounds,
        PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs,  PTSTREE:-PTNoNodess,
        PTSTREE:-PTBasess, PTSTREE:-PTdatas, w, slicesz, slicedims,
        Ssz, S, dVsz, pdegVec, cVsz, coefVec):

    slicedims[N+1]:=tmp:
    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
    #rep:-DGS:=(RecdenConnector:-PartialDegs(RPoly))[2]:
    rep:-DGS:=0:
    rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF:=0:
    rep:-CRep:-PDGSV:=pdegVec:
    rep:-CRep:-COEFV:=coefVec:
    rep:-CRepCase:=2:
    return rep:
end proc:
#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# {
# {
# Output:
# {
# {
#------------------------------------------------------------------------------
## start: the row (or index) at which to start seaching a subresultant
## whose LC is not zero, going bottom-up
## start from "start+1"-th row the bottom of the subresultant-chain,
## to trace the first valid row, i.e. whoes leading coefficient will
## notvanish.

FastInterpolateNextDefectiveLC:=proc(R, N, M, start, PTSTREE, w, Scube, p)
    local ESTIDGS, NMp, PHWD, S, Ssz, cVsz, coefVec, dVsz, i, j, pdegVec, rep,
    rep2, slicedims, slicesz, theend, tmp, ww, thestart;

    if (start = w-1) then
       rep2:=Record('ith', 'LC');
       rep2:-ith:=-1;
       rep2:-LC:=0:
       return rep2:
    end if:

    if(start<0) then thestart := 0: else thestart := start: end if:

    #for j from thestart+1 to w-1 do
    #   if(((Scube:-SLc)[j+1])<>0) then
    #      return (Scube:-SLc)[j+1]:
    #   end if:
    #end do:

    slicedims:=Scube:-Sdims:
    Ssz:=Scube:-Ssize:
    S:=Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := PTSTREE:-PTbounds[i] -1 end do;
    ESTIDGS[N+1] := 0:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    PHWD:=Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1]:=PTSTREE:-PTptssSz: PHWD[2]:=PTSTREE:-PThsSz:
    PHWD[3]:=PTSTREE:-PTWNBsSz: PHWD[4]:=PTSTREE:-PTdatasSz:
    ww:=w*w;
    slicesz:=iquo(Ssz,ww):
    tmp:=slicedims[N+1]:
    slicedims[N+1]:=1:

    theend:=ConnectorModuleNew1:-FastInterpNextDefectiveLCM(NMp, PHWD, start,
        PTSTREE:-PTbounds, PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs, 
        PTSTREE:-PTNoNodess, PTSTREE:-PTBasess, PTSTREE:-PTdatas, w, slicesz,
        slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec):

    slicedims[N+1]:=tmp:
    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    if theend = -1 then
      rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:=0;
      rep2:=Record('ith', 'LC');
      rep2:-ith:=theend;
      rep2:-LC:=rep:
    else
      rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
      #rep:-DGS:=(RecdenConnector:-PartialDegs(RPoly))[2]:
      rep:-DGS:=0:
      rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
      rep:-CRep:-COEF:=0:
      rep:-CRep:-PDGSV:=pdegVec:
      rep:-CRep:-COEFV:=coefVec:
      rep:-CRepCase:=2:
      rep2:=Record('ith', 'LC');
      rep2:-ith:=theend;
      rep2:-LC:=rep:
      #(Scube:-SLc)[(rep2:-ith)+1]:=rep2:
    end if:
    return rep2:
end proc:

#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# {
# {
# Output:
# {
# {
#------------------------------------------------------------------------------
## start: the row (or index) at which to start seaching a subresultant
## whose LC is not zero, going bottom-up
##  start from "start+1"-th row the bottom of the subresultant-chain,
##   to trace the first valid row, i.e. whoes leading coefficient will
##   notvanish.
FastInterpolateNextLC:=proc(R, N, M, start, PTSTREE, w, Scube, p)
    local ESTIDGS, NMp, PHWD, S, Ssz, cVsz, coefVec, dVsz, i, j, pdegVec,
    rep, rep2, slicedims, slicesz, theend, tmp, ww, thestart;

    if (start = w-1) then
       rep2:=Record('ith', 'LC');
       rep2:-ith:=-1;
       rep2:-LC:=0:
       return rep2:
    end if:
    
    if(start<0) then thestart := 0: else thestart := start: end if:
    
    for j from thestart+1 to w-1 do
       if(((Scube:-SLc)[j+1])<>0) then
          return (Scube:-SLc)[j+1]:
       end if:
    end do:
    
    slicedims:=Scube:-Sdims:
    Ssz:=Scube:-Ssize:
    S:=Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := PTSTREE:-PTbounds[i] -1 end do;
    ESTIDGS[N+1] := 0:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    PHWD:=Array(1..4, 'datatype' = 'integer[4]'):
    PHWD[1]:=PTSTREE:-PTptssSz: PHWD[2]:=PTSTREE:-PThsSz:
    PHWD[3]:=PTSTREE:-PTWNBsSz: PHWD[4]:=PTSTREE:-PTdatasSz:
    ww:=w*w;
    slicesz:=iquo(Ssz,ww):
    tmp:=slicedims[N+1]:
    slicedims[N+1]:=1:
    
    theend:=ConnectorModuleNew1:-FastInterpNextLCM(NMp, PHWD, start,
        PTSTREE:-PTbounds, PTSTREE:-PTptss, PTSTREE:-PThs, PTSTREE:-PTWs,
        PTSTREE:-PTNoNodess, PTSTREE:-PTBasess, PTSTREE:-PTdatas, w, slicesz,
        slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec):

    slicedims[N+1]:=tmp:
    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    if theend = -1 then
      rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:=0;
      rep2:=Record('ith', 'LC');
      rep2:-ith:=theend;
      rep2:-LC:=rep:
    else
      rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
      #rep:-DGS:=(RecdenConnector:-PartialDegs(RPoly))[2]:
      rep:-DGS:=0:
      rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
      rep:-CRep:-COEF:=0:
      rep:-CRep:-PDGSV:=pdegVec:
      rep:-CRep:-COEFV:=coefVec:
      rep:-CRepCase:=2:
      rep2:=Record('ith', 'LC');
      rep2:-ith:=theend;
      rep2:-LC:=rep:
      (Scube:-SLc)[(rep2:-ith)+1]:=rep2:
    end if:
    return rep2:
end proc:

## N: index of the main variable
## M: number the variables that we specialize (should N-1)
## Prepared the roots for the multidimensional-FFT for subresultant
## Prepared for subresultant chain computation
DftPreComp := proc(N, M, ININrep1, ININrep2, p)
    local mymsg, DftPreCompRes, bounds, dims1, dims2, es, i, m, rep1, rep2,
    roots, times, INrep1, INrep2, V;

    INrep1:=PartialDegsAssert(ININrep1):
    INrep2:=PartialDegsAssert(ININrep2):
    rep1:=CRepAssert(INrep1, 1):
    rep2:=CRepAssert(INrep2, 1):
    bounds:=Array(1..M+1, 'datatype' = 'integer[4]'):
    es:=Array(1..M+1, 'datatype' = 'integer[4]'):
    dims1:=Array(1..N+1, 'datatype' = 'integer[4]'):
    dims2:=Array(1..N+1, 'datatype' = 'integer[4]'):
    for i from M+2 to N+1 do
      dims1[i]:=(rep1:-DGS[i-1])+1:
      dims2[i]:=(rep2:-DGS)[i-1]+1:
    end do:

    ## WP09, variable list
    V := op(2, op(1, rep1[RecdenPoly]));

    ##print("rep1:-DGS=", rep1:-DGS);
    ##print("rep2:-DGS=", rep2:-DGS);

    for i from 2 to M+1 do
       userinfo(10, MODPN, "f = ", PolynomialConvertOut(rep1));
       userinfo(10, MODPN, "g = ", PolynomialConvertOut(rep2));
       userinfo(10, MODPN, "Variables", V);
       userinfo(10, MODPN, "Main variable", V[1]);
       userinfo(10, MODPN, "Target variable", V[-i+1]);

       ##bounds[i]:=(rep1:-DGS)[i-1]*(rep2:-DGS[N]) + (rep2:-DGS)[i-1]*(rep1:-DGS)[N] +1:
       bounds[i] := MinimalDegreeBound(PolynomialConvertOut(rep1), PolynomialConvertOut(rep2), V[1], V[-i+1]) + 1;

       userinfo(8, MODPN, "WAS", (rep1:-DGS)[i-1]*(rep2:-DGS[N]) + (rep2:-DGS)[i-1]*(rep1:-DGS)[N] +1);
       userinfo(8, MODPN, "NOW", bounds[i]);

       es[i]:=log2Ceil(bounds[i]);
       dims1[i]:=2^(es[i]):
       dims2[i]:=dims1[i]:
    end do:
    m:=0;
    for i from 2 to M+1 do m:=m+dims1[i]; end do;
    if(m > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    roots:=Array(1..m, 'datatype' = 'integer[4]'):

    times:=ConnectorModuleNew1:-createGoodRoots(N, M, rep1:-DGS, rep1:-CRep:-COEF, rep2:-DGS, rep2:-CRep:-COEF, es, dims1, roots, p):

    mymsg:=times:

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
     if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    if (times > 10) then
     return 0:
    end if:
    for  i from 2 to M+1 do bounds[i]:=dims1[i]: end do:
    DftPreCompRes:=Record( 'TIMES', 'ES', 'BOUNDS', 'DIMS1', 'DIMS2', 'ROOTS'):
    DftPreCompRes:-ES:=es:
      ## Fourier degrees that we need (in each dimension)
    DftPreCompRes:-BOUNDS:=bounds:
      ## see code
    DftPreCompRes:-DIMS1:=dims1:
      ## FFT size
    DftPreCompRes:-DIMS2:=dims2:
      ## FFT size
    DftPreCompRes:-ROOTS:=roots:
      ## the prmitive roots (one in each dimension)
    DftPreCompRes:-TIMES:=times:
    return DftPreCompRes:
end proc;

## Similar to above but for division
DftPreCompDiv := proc(N, M, ININrep1, ININrep2, p)
    local DftPreCompRes, bounds, dims1, dims2, es, i, m, rep1, rep2, roots,
    times, INrep1, INrep2;

    INrep1:=PartialDegsAssert(ININrep1):
    INrep2:=PartialDegsAssert(ININrep2):
    rep1:=CRepAssert(INrep1, 1):
    rep2:=CRepAssert(INrep2, 1):
    bounds:=Array(1..M+1, 'datatype' = 'integer[4]'):
    es:=Array(1..M+1, 'datatype' = 'integer[4]'):
    dims1:=Array(1..N+1, 'datatype' = 'integer[4]'):
    dims2:=Array(1..N+1, 'datatype' = 'integer[4]'):
    for i from M+2 to N+1 do
      dims1[i]:=(rep1:-DGS[i-1])+1:
      dims2[i]:=(rep2:-DGS)[i-1]+1:
    end do:

    for i from 2 to M+1 do
       bounds[i]:=(rep1:-DGS)[i-1] + ((rep1:-DGS)[N] - (rep2:-DGS)[N]) * (rep2:-DGS)[i-1]:
       es[i]:=log2Ceil(bounds[i]*2);
       dims1[i]:=2^(es[i]):
       dims2[i]:=dims1[i]:
    end do:

    m:=0;
    for i from 2 to M+1 do m:=m+dims1[i]; end do;
    if(m > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    roots:=Array(1..m, 'datatype' = 'integer[4]'):

    times:=ConnectorModuleNew1:-createGoodRoots(N, M, rep1:-DGS, rep1:-CRep:-COEF, rep2:-DGS, rep2:-CRep:-COEF, es, dims1, roots, p):

    if ((times > 10)  or (times = -1)) then return 0: end if:
    for  i from 2 to M+1 do bounds[i]:=dims1[i]: end do:
    DftPreCompRes:=Record( 'TIMES', 'ES', 'BOUNDS', 'DIMS1', 'DIMS2', 'ROOTS'):
    DftPreCompRes:-ES:=es:
    DftPreCompRes:-BOUNDS:=bounds:
    DftPreCompRes:-DIMS1:=dims1:
    DftPreCompRes:-DIMS2:=dims2:
    DftPreCompRes:-ROOTS:=roots:
    DftPreCompRes:-TIMES:=times:
    return DftPreCompRes:
end proc;

## N: index of the main variable
## M: number the variables that we specialize (should N-1)
## ININrep: polynomial
## es: array of Fourier degrees
## dims: array of FFT sizes
## array of primitive roots from least variable (index = 1)
##                          to top evaluated variable (index = M)
##  Returns ECube
DftEvaluationM:=proc(N, M, ININrep, es, dims, ROOTS, p)
    local Ecube, Edat, Esz, NMp, i, pSz, rep, INrep;

    INrep:=PartialDegsAssert(ININrep):
    rep:=CRepAssert(INrep, 1):
    pSz:=calSizDgs(rep:-DGS,N):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    Esz:=1:
    for i from 2 to N+1 do
        Esz:=Esz*(dims)[i]:
    end do:
    if(Esz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    Edat:=Array(1..Esz, 'datatype' = 'integer[4]'):

    ConnectorModuleNew1:-DftEvalM (NMp, es, dims, Esz, Edat, rep:-DGS, pSz, rep:-CRep:-COEF, ROOTS):
    Ecube:=Record( 'Ees', 'Edims', 'Esize', 'Edata'):
    Ecube:-Ees:=es:
    Ecube:-Edims:=dims:
    Ecube:-Esize:=Esz:
      ## size of Edat:
    Ecube:-Edata:=Edat:
      ## data array, its first position is 1.
    return Ecube:
end proc:

#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# { p : prime number, the characteristic of the field }
# { N : the total number of variables appearing }
# { M : the number of variables to be evaluated }
# { R : a modpn polynomial ring of characteristic `p` and with
# variable list }
# { PTSTREE : sub-product tree data-structure. }
# { ESTIDGS : esimate partial degrees for the output polynomial }
# { Edims, Esz, E : output of the evaluation }
# Output:
# { a modpn multivariate polynomial in recden format }
#------------------------------------------------------------------------------
## same as InterpolationM(), except using DFT..

DftInterpolationM:=proc(R, N, M, ESTIDGS, Ecube, ROOTS, p)
    local NMp, cVsz, coefVec, dVsz, pdegVec, rep;
    dVsz:=estimatePartialDegVecSize(ESTIDGS, N)+1:
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgs(ESTIDGS, N) + 1:
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    ConnectorModuleNew1:-DftInterpM(NMp, Ecube:-Ees, Ecube:-Edims, Ecube:-Esize, Ecube:-Edata, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
    #rep:-DGS:=(RecdenConnector:-PartialDegs(rep:-RecdenPoly))[2]:
    rep:-DGS:=0:
    rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF:=0:
    rep:-CRep:-PDGSV:=pdegVec:
    rep:-CRep:-COEFV:=coefVec:
    rep:-CRepCase:=2:
    return rep:
end proc:

## same as InterpolationPolynomial(), except using DFT..
DftInterpolatePolynomial:=proc(R, N, M, ith, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep, slicedims, slicees, slicesz;

    if(((Scube:-SPoly)[ith+1]) <> 0) then
         return (Scube:-SPoly)[ith+1]:
    end if:

    slicees:=Scube:-Ses:
    slicedims:=Scube:-Sdims:
    Ssz:=Scube:-Ssize:
    S:=Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] -1 end do:
    ESTIDGS[N+1] := w-1:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    slicesz:=iquo(Ssz,w):
    ConnectorModuleNew1:-DftInterpithM(NMp, ith, w, slicesz,  slicees,slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
    rep:-DGS:=0:
    rep:-CRep := 0:
    rep:-CRepCase:=-1:

    if(nargs=9) then rep:=ReduceCoeffs(R, rep, args[9]): end if:

    (Scube:-SPoly)[ith+1]:=rep:
    return rep:
end proc:

DftInterpolateCoefficient:=proc(R, N, M, ith, dth, w, Scube, ROOTS,  p)
     local ESTIDGS, NMp, S, Ssz, cVsz, coefVec, dVsz, i, pdegVec, rep, slicedims, slicees, slicesz, tmp, ww;
     slicees:=Scube:-Ses:
     slicedims:=Scube:-Sdims:
     Ssz:=Scube:-Ssize:
     S:=Scube:-Sdata:
     ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
     for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] -1 end do;
     ESTIDGS[N+1] := 0:
     dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
     if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
     pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
     cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
     if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
     coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
     NMp:=Array(1..3, 'datatype' = 'integer[4]'):
     NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
     ww:=w*w;
     slicesz:=iquo(Ssz,ww):
     tmp:=slicedims[N+1]:
     slicedims[N+1]:=1:
     ConnectorModuleNew1:-DftInterpithdthM(NMp, ith, dth, w,  slicesz, slicees,slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):
     slicedims[N+1]:=tmp:

     rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):

     rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
     #rep:-DGS:=(RecdenConnector:-PartialDegs(rep:-RecdenPoly))[2]:
     rep:-DGS:=0:
     rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
     rep:-CRep:-COEF:=0:
     rep:-CRep:-PDGSV:=pdegVec:
     rep:-CRep:-COEFV:=coefVec:
     rep:-CRepCase:=2:
     return rep:
end proc:

## The same as nterpolateNextLC(), except using DFT..
DftInterpolateNextDefectiveLC:=proc(R, N, M, start, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz,cVsz, coefVec, dVsz, i, j, pdegVec, rep, rep2,
    slicedims, slicees, slicesz, theend, tmp, ww, thestart;

    if start = w-1 then
       rep2:=Record('ith', 'LC');
       rep2:-ith:=-1;
       rep2:-LC:=0:
       return rep2:
    end if:

    if (start<0) then thestart := 0: else thestart := start: end if:


    #for j from thestart+1 to w-1 do
    #   if(((Scube:-SLc)[j+1])<>0) then
    #      return (Scube:-SLc)[j+1]:
    #   end if:
    #end do:

    slicees:=Scube:-Ses:
    slicedims:=Scube:-Sdims:
    Ssz:=Scube:-Ssize:
    S:=Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] -1 end do;
    ESTIDGS[N+1] := 0:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    ww:=w*w;
    slicesz:=iquo(Ssz,ww):
    tmp:=slicedims[N+1]:
    slicedims[N+1]:=1:
    theend:=ConnectorModuleNew1:-DftInterpNextDefectiveLCM(NMp, start, w,
        slicesz, slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    slicedims[N+1]:=tmp:

    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):

    if theend = -1 then
      rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:=0;
      rep2:=Record('ith', 'LC');
      rep2:-ith:=theend;
      rep2:-LC:=rep:
    else
      rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
      #rep:-DGS:=(RecdenConnector:-PartialDegs(rep:-RecdenPoly))[2]:
      rep:-DGS:=0:
      rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
      rep:-CRep:-COEF:=0:
      rep:-CRep:-PDGSV:=pdegVec:
      rep:-CRep:-COEFV:=coefVec:
      rep:-CRepCase:=2:
      rep2:=Record('ith', 'LC');
      rep2:-ith:=theend;
      rep2:-LC:=rep:
      #(Scube:-SLc)[(rep2:-ith) + 1]:=rep2:
    end if:
    return rep2:
end proc:

## The same as nterpolateNextLC(), except using DFT..
DftInterpolateNextLC:=proc(R, N, M, start, w, Scube, ROOTS, p)
    local ESTIDGS, NMp, S, Ssz,cVsz, coefVec, dVsz, i, j, pdegVec, rep,
    rep2, slicedims, slicees, slicesz, theend, tmp, ww, thestart;

    if start = w-1 then
       rep2:=Record('ith', 'LC');
       rep2:-ith:=-1;
       rep2:-LC:=0:
       return rep2:
    end if:
    
    if (start<0) then thestart := 0: else thestart := start: end if:
    
    for j from thestart+1 to w-1 do
       if(((Scube:-SLc)[j+1])<>0) then return (Scube:-SLc)[j+1]: end if:
    end do:
    
    slicees:=Scube:-Ses:
    slicedims:=Scube:-Sdims:
    Ssz:=Scube:-Ssize:
    S:=Scube:-Sdata:
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := slicedims[i] -1 end do;
    ESTIDGS[N+1] := 0:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
    NMp:=Array(1..3, 'datatype' = 'integer[4]'):
    NMp[1]:=N: NMp[2]:=M: NMp[3]:=p:
    ww:=w*w;
    slicesz:=iquo(Ssz,ww):
    tmp:=slicedims[N+1]:
    slicedims[N+1]:=1:
    
    theend:=ConnectorModuleNew1:-DftInterpNextLCM(NMp, start, w, slicesz,
        slicees, slicedims, Ssz, S, dVsz, pdegVec, cVsz, coefVec, ROOTS):
    slicedims[N+1]:=tmp:
    
    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    if theend = -1 then
      rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:=0;
      rep2:=Record('ith', 'LC');
      rep2:-ith:=theend;
      rep2:-LC:=rep:
    else
      rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
      rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, pdegVec, coefVec, R:-VarList):
      #rep:-DGS:=(RecdenConnector:-PartialDegs(rep:-RecdenPoly))[2]:
      rep:-DGS:=0:
      rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
      rep:-CRep:-COEF:=0:
      rep:-CRep:-PDGSV:=pdegVec:
      rep:-CRep:-COEFV:=coefVec:
      rep:-CRepCase:=2:
      rep2:=Record('ith', 'LC');
      rep2:-ith:=theend;
      rep2:-LC:=rep:
      (Scube:-SLc)[(rep2:-ith) + 1]:=rep2:
    end if:
    return rep2:
end proc:

## To estimate the size of the buffers should be allocated for
# the output for the Lifitng algorithm.
## THis is for the lifting of a variable.
## The estiamtes are for a 2-Vec representation
## N: the number of variables.
## TriSetDgs: the partial degrees of the triangular for the Lifting algorithm.
## m: the index of the variable who is to be lifted.
## deg: in the Lifting algorithm we want to lift x_m to x_m^deg.
estimateForOnePoly_Lift:=proc(N, TriSetDgs, m, deg)
    local ESTIDGS, cVsz, dVsz, i;
    ESTIDGS:=Array(1..N+1, 'datatype' = 'integer[4]'):
    for i from 1 to N do
      ESTIDGS[i+1]:=TriSetDgs[i]:
    end do:
    if (m <= N) then
      ESTIDGS[m+1]:=deg:
    end if:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
    return [dVsz, cVsz]:
end proc:

## R: the MODPN ring.
## M: 0<= M <=N
## N: the number of variables.
## InTriSet: a triangular set.
## return the paritial degrees for InTriSet from X1 to XM.
getTriSetDegs := proc(R, M, N, InTriSet)
      local TriSetDgs, i;
      TriSetDgs:=Array(1..M, 'datatype' = 'integer[4]'):
      for i from 1 to M do
         if (InTriSet[i]<>0) then
           TriSetDgs[i]:=Algebraic:-RecursiveDensePolynomials:-degrpoly(InTriSet[i]:-RecdenPoly, (R:-VarList)[N-i+1]):
         end if:
      end do:
      return TriSetDgs:
end proc:

## R: the MODPN ring.
## X_Y: X_Y is the variable will be lifted.
## InSystem: the input system for the Lifting algorithm.
## InTriSet: the input triangular set.
## estimate the C-2Vector representation for the lifting algorithm...
## the order of the triangular set from smallest to biggest
# one
## Both InSystem and InTriSet are lists of modpn polynomials
## For the InTriSet polynomials are sorted from the smallest
estimateThe2VecSiz_Lift:=proc(R, X_Y, InSystem, InTriSet)
    local N, TriSetDgs, VarSet, coefSiz, deg, i, m, no, pdegSiz, rec;
    N:=nops(R:-VarList):
    no:=nops(InSystem):
    deg:=1:
    pdegSiz:=0;
    coefSiz:=0;
    VarSet := convert(R:-VarList,set):
    m:=0:

    for i from 1 to no do
      deg:=deg*degree(InSystem[i], VarSet):
    end do:
    deg:=2*deg:
    for i from 1 to N do
        if((R:-VarList)[i] = X_Y ) then
           m:=i:
        end if:
    end do:

    TriSetDgs:=Array(1..N, 'datatype' = 'integer[4]'):

    m:=N-m+1:

    for i from 1 to m-1 do
      TriSetDgs[i]:=Algebraic:-RecursiveDensePolynomials:-degrpoly(InTriSet[i]:-RecdenPoly, (R:-VarList)[N-i+1]):
    end do:
    for i from m+1 to N do
      TriSetDgs[i]:=Algebraic:-RecursiveDensePolynomials:-degrpoly(InTriSet[i-1]:-RecdenPoly, (R:-VarList)[N-i+1]):
    end do:
    for i from 1 to m-1 do
        rec:=estimateForOnePoly_Lift(i, TriSetDgs, 1, deg):
        pdegSiz:=pdegSiz+rec[1]:
        coefSiz:=pdegSiz+rec[2]:
    end do:
    for i from m+1 to N do
        rec:=estimateForOnePoly_Lift(i, TriSetDgs, 1, deg):
        pdegSiz:=pdegSiz+rec[1]:
        coefSiz:=pdegSiz+rec[2]:
    end do:

    for i from m-1 by -1 to 1 do
        TriSetDgs[i+1]:=TriSetDgs[i]:
    end do:

    TriSetDgs[1]:=1:

    return [pdegSiz, coefSiz, TriSetDgs]:
end proc;

## return the index of variable 'X_Y' in the 'VarList'.
varno :=proc(VarList, X_Y)
   local  N, Y, i;
   Y:=0:
   N:=nops(VarList):
   for i from 1 to N do
        if(VarList[i] = X_Y) then
           Y:=i:
        end if:
   end do:
   return Y:
end proc:




## InSystem: the input system.
## VarList: the variable list.
## To merge input system (a list of polynomials) into arrays
##   so easier to pass to the C level.
convertAndMergeSLG_Lift := proc(InSystem, VarList)
   local A, GNS, L, MDAS, MDASSZ, i, j, k, n, no, offset;
   L:=[]:
   no:=nops(InSystem):
   for i from 1 to no do
     A:=MapleDagModule:-MapleDag2CData(InSystem[i], VarList):
     L:=[ op(L), A]:
   end do:
   n:=nops(L):
   GNS:=Array(1..n, 'datatype'='integer[4]'):
   MDASSZ:=0:
   for i from 1 to n do
     GNS[i]:=op(op(L[i])[1])[2]:
     MDASSZ := MDASSZ + GNS[i]*3:
   end do:
   if(MDASSZ > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   MDAS:=Array(1..MDASSZ, 'datatype'='integer[4]'):
   offset:=0:
   for i from 1 to n do
     for j from 1 to GNS[i] do
       for k from 1 to 3 do
           MDAS[offset+(j-1)*3+k]:=L[i][j][k]:
       end do:
     end do:
     offset:=offset+GNS[i]*3:
   end do:
   return [GNS, MDAS]:
end proc:

## InTriSet: the Triangular Set for lifting.
## To assert the C representation exist.
CRepAssertTriSet_Lift := proc(InTriSet)
    local  L, i, no, rep, inrep;
    L:=[]:
    no:=nops(InTriSet):
    for i from 1 to no do
       inrep:=CRepAssert(InTriSet[i], 1):
       rep:=PartialDegsAssert(inrep):
       L:=[rep, op(L)]:
    end do:
    return L:
end proc:


## Suppose the first one is y.
## N: a number less or equal to the number of polynomials in 'InTriSet'.
## InTriSet: the input triangular set.
## To merge the polynomails in the input triangular into arrays,
##    these arrays will be passed to C level.
MergeTriSetRep_Lift := proc(N, InTriSet)
   local  COEFSZ, NewInTriSet, dgno, i, inCOEFS, inDGS, inSIZS, j, no, offset, offset2;
   #print("N=", N);
   #print("InTriSet=", op(InTriSet));
   NewInTriSet:=CRepAssertTriSet_Lift(InTriSet):
   NewInTriSet:=ListTools:-Reverse(NewInTriSet):
   inSIZS:=Array(1..N, 'datatype'='integer[4]'):
   dgno:=N*N:
   no:=nops(NewInTriSet):
   inDGS:= Array(1..dgno, 'datatype'='integer[4]'):
   inSIZS[1]:=2: # [0,1]
   inDGS[1]:=1:
   offset:=N:
   COEFSZ:=inSIZS[1]:

   #print("NewInTriSet=", op(NewInTriSet));

   for i from 1 to no do
     #print("00---i---", i):
     #print("op(NewInTriSet[i]) = ", NewInTriSet[i]):
     inSIZS[i+1]:=ArrayNumElems((NewInTriSet[i]):-CRep:-COEF):
     for j from 1 to ArrayNumElems(NewInTriSet[i]:-DGS) do
         #print("01---i---", i):
         inDGS[offset+j]:=(NewInTriSet[i]:-DGS)[j]:
     end do:
     offset:=offset+N:
     COEFSZ:=COEFSZ+inSIZS[i+1]:
   end do:


   inCOEFS:= Array(1..COEFSZ, 'datatype'='integer[4]'):
   inCOEFS[1]:=0:
   inCOEFS[2]:=1:
   offset2:=2:
   for i from 1 to no do
     for j from 1 to inSIZS[i+1] do
         #print("02---i---", i):
         inCOEFS[offset2+j]:=((NewInTriSet[i]):-CRep:-COEF)[j]:
     end do:
     offset2:=offset2+inSIZS[i+1]:
   end do:

   return [inDGS, inSIZS, inCOEFS]:
end proc:

## R: the modpn ring.
## TriSet: the input triangular set.
## N: is number less or equal to the number inside the triangular set 'TriSet'.
## Y: the index of the variable to be lifted in the lifitng algorithm.
## This function exchanges the variable X_Y with X_1 for the polynomials in the triangular set 'TriSet'.
MapTriSetX_Y2ONE_Lift_IN:=proc(R, TriSet, N, Y)
      local MTriSet, finalTriSet, i, j, newTriSet, newpoly, no;
      no:=nops(TriSet):
      MTriSet:=[]:
      for i from 1 to no do
            MTriSet:=[op(MTriSet),  PolynomialConvertOut(TriSet[i])]:
      end do:
      newTriSet:=[]:
      for i from 1 to no do
        newpoly:=MTriSet[i]:
        for j from Y+1 to N do
           newpoly:=subs((R:-VarList)[j]= (R:-VarList)[j-1], newpoly):
        end do:
        newTriSet:=[op(newTriSet), newpoly]:
      end do:
      finalTriSet:=[]:
      for i from 1 to no do
            finalTriSet:=[op(finalTriSet), PolynomialConvertIn(R, newTriSet[i])]:
      end do:
      return finalTriSet:
end proc:

## R: the modpn ring.
## TriSet: the input triangular set.
## N: is number less or equal to the number inside the triangular set 'TriSet'.
## Y: the index of the variable to be lifted in the lifitng algorithm.
## This function exchanges the variable X_1 with X_Y for the polynomials in the triangular set 'TriSet'.
MapTriSetX_Y2ONE_Lift_OUT:=proc(R, TriSet, N, Y)
      local MTriSet, finalTriSet, i, j, newTriSet, newpoly, no;
      no:=nops(TriSet):
      MTriSet:=[]:
      for i from 1 to no do
            MTriSet:=[op(MTriSet),  PolynomialConvertOut(TriSet[i])]:
      end do:
      newTriSet:=[]:
      for i from 1 to no do
        newpoly:=MTriSet[i]:
        newpoly:=subs((R:-VarList)[N]=tmpVar, newpoly):
        for j from N-1 by -1 to Y do
           newpoly:=subs((R:-VarList)[j]= (R:-VarList)[j+1], newpoly):
        end do:
        newpoly:=subs(tmpVar=(R:-VarList)[Y], newpoly):
        newTriSet:=[op(newTriSet), newpoly]:
      end do:
      finalTriSet:=[]:
      for i from 1 to no do
            finalTriSet:=[op(finalTriSet), PolynomialConvertIn(R, newTriSet[i])]:
      end do:
      return finalTriSet:
end proc:

## R: the MODPN ring.
## no: dedicate the number of polynomial should be interprated from outPDGVECS, outCOEFVECS.
## outPDGVECS: the paritial degree vector for the polynomials of the output triangular set from the Lifting algorithm.
## outCOEFVECS: the coefficient vector for the polynomials of the output triangular set from the Lifting algorithm.
## This function decodes the output from the C level Lifting algorithm.
## This function will return a list of Modpn polynomails which
# is the output triangular set for the Lifting algorithm.
## This function is the reverse of the above "merge"
splitTriSetRep_Lift:=proc(R, no, outPDGVECS, outCOEFVECS, p)
     local COEFVEC, N, PDGVEC, TriSet, VarL, i, j, offset1, offset2, rep, sz1, sz2;
     TriSet:=[]:
     offset1:=0:
     offset2:=0:
     N:=nops(R:-VarList):
     for i from 1 to no do
         sz1:=outPDGVECS[offset1+1]+1:
         if(sz1 > MEMORY_BOUND)  then error ERR_MEMORY: end if:
         PDGVEC:=Array(1..sz1, 'datatype'='integer[4]'):
         for j from 1 to sz1 do
              PDGVEC[j]:=outPDGVECS[offset1+j]:
         end do:
         offset1:=offset1+sz1:
         sz2:=outCOEFVECS[offset2+1]+1:
         if(sz2 > MEMORY_BOUND)  then error ERR_MEMORY: end if:
         COEFVEC:=Array(1..sz2, 'datatype'='integer[4]'):
         for j from 1 to sz2 do
              COEFVEC[j]:=outCOEFVECS[offset2+j]:
         end do:
         offset2:=offset2+sz2:
         rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
         VarL:=[]:
         for j from N by -1 to N-i do
           VarL:=[(R:-VarList)[j], op(VarL)]:
         end do:
         rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, PDGVEC, COEFVEC, VarL):
         rep:-DGS:=0:
         rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
         rep:-CRep:-COEF:=0:
         rep:-CRep:-PDGSV:=PDGVEC:
         rep:-CRep:-COEFV:=COEFVEC:
         rep:-CRepCase:=2:
         TriSet:=[op(TriSet), rep]:
     end do:
     return TriSet:
end proc:

# when specializing the TriSet.
# InSystem: a list of Maple Polynomial.
# InTriSet: a list of Modpn Polynomial.
# X_Y: the variable has been specialized.
# y0: the valuation point.
# Returns a list of modpn polynomials
HenselLifting:=proc(R, InSystem, InTriSet, X_Y, y0)
    local N, NewInTriSet, NewY, Y, itertimes, no, outCOEFVECS, outPDGVECS,
    outTriSet, rec1, rec2, rec3;

    N := nops(R:-VarList):
    rec1:=estimateThe2VecSiz_Lift(R, X_Y, InSystem, InTriSet):
    if(rec1[1] > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    if(rec1[2] > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    outPDGVECS:=Array(1..rec1[1], 'datatype'='integer[4]'):
    outCOEFVECS:=Array(1..rec1[2], 'datatype'='integer[4]'):
    Y:=varno(R:-VarList, X_Y):

    NewInTriSet:=MapTriSetX_Y2ONE_Lift_IN(R, InTriSet, N, Y):

    rec2:=convertAndMergeSLG_Lift(InSystem, R:-VarList):
    rec3:=MergeTriSetRep_Lift(N, NewInTriSet):
    NewY:=N-Y+1;

    itertimes:=ConnectorModuleNew1:-HenselLiftingC(outPDGVECS, outCOEFVECS, NewY,
        y0, N, rec2[1], rec2[2], rec1[3], rec3[1], rec3[2], rec3[3], R:-prime):
    if(itertimes = -1) then
        return 0:
    end if:

    outTriSet:=[]:
    no:=nops(InTriSet):
    outTriSet:=splitTriSetRep_Lift(R, no, outPDGVECS, outCOEFVECS, R:-prime):

    outTriSet:=MapTriSetX_Y2ONE_Lift_OUT(R, outTriSet, N, Y):
    return outTriSet:
end proc:

CRepAssertTriSet := proc(no, InTriSet)
    local  L, i, inrep;
    L:=[seq(0, i = 1 ..no )]:
    for i from 1 to no do
      if (InTriSet[i] <> 0) then
         inrep:=CRepAssert(InTriSet[i], 1):
         L := subsop(i=PartialDegsAssert(inrep), L):
         #L[i]:=PartialDegsAssert(inrep):
      end if:
    end do:
    return L:
end proc:

## Pack the poly data before passing it to C
MergeTriSetRep := proc(no, InTriSet)
   local COEFSZ, NewInTriSet, dgno, i, inCOEFS, inDGS, inSIZS, j, offset, offset2;
   NewInTriSet:=CRepAssertTriSet(no, InTriSet):
   if(no > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   inSIZS:=Array(1..no, 'datatype'='integer[4]'):
   dgno:=no*no:
   inDGS:= Array(1..dgno, 'datatype'='integer[4]'):

   offset:=0:
   COEFSZ:=0:

   for i from 1 to no do
      if(NewInTriSet[i]<>0) then
         inSIZS[i]:=ArrayNumElems((NewInTriSet[i]):-CRep:-COEF):
         for j from 1 to no do
            inDGS[offset+j]:=(NewInTriSet[i]:-DGS)[j]:
         end do:
      end if:
      offset:=offset+no:
      COEFSZ:=COEFSZ+inSIZS[i]:
   end do:

   if(COEFSZ > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   inCOEFS:= Array(1..COEFSZ, 'datatype'='integer[4]'):
   offset2:=0:
   for i from 1 to no do
      if(NewInTriSet[i]<>0) then
        for j from 1 to inSIZS[i] do
         inCOEFS[offset2+j]:=((NewInTriSet[i]):-CRep:-COEF)[j]:
        end do:
      end if:
     offset2:=offset2+inSIZS[i]:
   end do:
   return [inDGS, inSIZS, inCOEFS]:
end proc:

getIndexOfVar := proc(Vars, var)
    local M, N, i:
    N:=nops(Vars):
    M:=0:
    for i from 1 to N do
       M:=i:
       if (Vars[i] = var) then break: end if:
    end do:
    return N-M+1:
end proc:


# return 1 means YES invertable.
# return 0 means NO.
# return -1 means InRep NOT reducded w.r.t InTriSet.
IsInvertable :=proc(R, Mvar, InInRep, InTriSet)
    local mymsg, N, NewRep, NewTriSet, TriSetDGS, i, rec3, res, M, InRep;
    #  Mvar is not used

    N:=nops(R:-VarList):
    M:=getIndexOfVar(R:-VarList, Mvar):
    # x_M is the mainvar for InRep.
    # x_N is the mainver for InTriSet.
    InRep:=PartialDegsAssert(InInRep):
    NewRep:=CRepAssert(InRep, 1):
    NewTriSet:=[seq(0, i = 1 .. M)]:
    for i from 1 to M do
        NewTriSet[i] := InTriSet[i]:
    end do:
    rec3:=MergeTriSetRep(M, NewTriSet):
    TriSetDGS:=getTriSetDegs(R, M, N, NewTriSet):
    res:=ConnectorModuleNew1:-IsInvertableCN(M, NewRep:-DGS, NewRep:-CRep:-COEF,
        TriSetDGS, rec3[1], rec3[2], rec3[3], R:-prime):

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    if (res=0) then
       return 0:
    else
       return 1:
    end if:
end proc:


## mulnipulate the Recden polynomail encoding,
## make the input Recdenpoly with M variable becomes one with N variables.
## But the input and output keep the same rest informations.
EnlargeRecden2N := proc(N, M, newvars, InRep)
    local coeffs, i, incr, newrop, rop;
    rop:=op(InRep:-RecdenPoly):
    incr:=N-M:
    coeffs:=rop[2]:
    for i from 1 to incr do
         coeffs:=[coeffs]:
    end do:
    newrop:=[rop[1][1], newvars, []], coeffs:
    return POLYNOMIAL(newrop):
end proc:

## mulnipulate the Recden polynomail encoding,
## make the input Recdenpoly with M variable (origianl number of variables)
## becomes one with N variables (the number of variable in newvars).
## But the input and output keep the same rest informations.
EnlargePoly :=proc (newvars, InRep)
    local  M, N, Rec, i, rop;
    rop:=op(InRep:-RecdenPoly):
    N:=nops(newvars):
    M:=nops(rop[1][2]):
    Rec:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    Rec:-DGS:=Array(1..N, 'datatype'='integer[4]'):
    for i from 1 to M do
         (Rec:-DGS)[i]:=(InRep:-DGS)[i]:
    end do:
    Rec:-RecdenPoly:=EnlargeRecden2N(N, M, newvars, InRep):
    Rec:-CRep:=0:
    Rec:-CRepCase:=-1:
    return Rec:
end proc:

## Unpack
splitTriSetRep:=proc(R, no, outPDGVECS, outCOEFVECS, p)
   local COEFVEC, N, PDGVEC, TriSet, VarL, i, j, offset1, offset2, rep, sz1, sz2;
   TriSet:=[]:
   offset1:=0:
   offset2:=0:
   N:=nops(R:-VarList):

   for i from 1 to no do
       sz1:=outPDGVECS[offset1+1]+1:
       if(sz1 > MEMORY_BOUND)  then error ERR_MEMORY: end if:
       PDGVEC:=Array(1..sz1, 'datatype'='integer[4]'):
       for j from 1 to sz1 do
            PDGVEC[j]:=outPDGVECS[offset1+j]:
       end do:
       offset1:=offset1+sz1:
       sz2:=outCOEFVECS[offset2+1]+1:
       if(sz2 > MEMORY_BOUND)  then error ERR_MEMORY: end if:
       COEFVEC:=Array(1..sz2, 'datatype'='integer[4]'):
       for j from 1 to sz2 do
            COEFVEC[j]:=outCOEFVECS[offset2+j]:
       end do:
       offset2:=offset2+sz2:
       rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
       VarL:=[]:
       for j from N by -1 to N-i+1 do
         VarL:=[(R:-VarList)[j], op(VarL)]:
       end do:
       rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(p, PDGVEC, COEFVEC, VarL):
       rep:-DGS:=0:
       rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
       rep:-CRep:-COEF:=0:
       rep:-CRep:-PDGSV:=PDGVEC:
       rep:-CRep:-COEFV:=COEFVEC:
       rep:-CRepCase:=2:
       TriSet:=[op(TriSet), rep]:
   end do:
   return TriSet:
end proc:

# The input Triangular Set should start from the Bottom upto the Top.
# R is the ring contains all the variables.
# InInRep's mvar may be smaller than the one in InTriSet.
# Mvar is the main variable InInRep.
# freeVar is the variable which is not algebraic in InInTriSet
# assumption:  InTriSet[getIndexOfVar(R:-VarList, freeVar)] is
# 0. (This checks that  freeVar is really free)

IteratedResultantOneDim :=  proc(R, Mvar, InInRep, InInTriSet, bound, freeVar)
    local mymsg, themvar, N, M, freeVarNo, InRep, NewRep, InTriSet, i, rec3,
    TriSetDGS, newbound, outVec, rep;

    N:=nops(R:-VarList):
    M:=getIndexOfVar(R:-VarList, Mvar):
    
    freeVarNo:=getIndexOfVar(R:-VarList, freeVar):
    InRep := PartialDegsAssert (InInRep):
    NewRep:=CRepAssert(InRep, 1):
    
    InTriSet:=[seq(0, i = 1 .. N)]:
    
    for i from 1 to freeVarNo-1 do
       InTriSet[i]:=InInTriSet[i]:
    end do:
    InTriSet[freeVarNo] := 0;
    for i from freeVarNo to N-1 do
       InTriSet[i+1] := InInTriSet[i]:
    end do:
    
    #print("N=", N);
    #print("InTriSet[1]=", InTriSet[1]):
    #print("InTriSet[2]=", InTriSet[2]:-RecdenPoly):
    rec3:=MergeTriSetRep(N, InTriSet):
    
    
    TriSetDGS:=getTriSetDegs(R, N, N, InTriSet):
    newbound := 2*(2*bound+1):
    if(newbound > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    outVec:=Array(1..newbound, 'datatype'='integer[4]'):
    mymsg:=ConnectorModuleNew1:-IterResultantOneDimCN(outVec, M, NewRep:-DGS,
        NewRep:-CRep:-COEF, N, TriSetDGS, rec3[1], rec3[2], rec3[3],
        bound, freeVarNo,  R:-prime);

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110) then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    rep:=Record('RecdenPoly', 'CRep'):
    rep:-RecdenPoly:=RecdenConnector:-C2RecdenPolyUni(R:-prime, outVec, (R:-VarList)[N]):
    rep:-CRep:=Record('DEG', 'COEFV'):
    rep:-CRep:-DEG:=bound:
    rep:-CRep:-COEFV:=outVec:
    rep:-CRep:=removeLeadingZeros(rep:-CRep):
    return rep:
end proc:

# cut into N pieces, each pieces constains M fragments.
getAListOfVectors := proc (InVec, N, M)
    local res, base, size, offset, i, j, Vec;
    res:=[]:
    base:=0:
    size:=1:
    offset:=1:

    for j from 1 to N do
       for i from 1 to M do
          offset:=offset + InVec[offset]+1;
       end do:
       size:=offset-size:
       if(size > MEMORY_BOUND)  then error ERR_MEMORY: end if:
       Vec:=Array(1..size, 'datatype'='integer[4]'):

       for i from 1 to size do
           Vec[i]:=InVec[base+i]:
       end do:

       base:=base+size:
       res:=[op(res), Vec]:
       size:=offset:
   end do:
   return res:
end proc:


## Builds a modpn polynomail from a 2-Vec polynomail.
Vec2Rep := proc (R, pdegVec, coefVec)
    local rep;
    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(R:-prime, pdegVec, coefVec, R:-VarList):
    rep:-DGS:=0:
    rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF:=0:
    rep:-CRep:-PDGSV:=pdegVec:
    rep:-CRep:-COEFV:=coefVec:
    rep:-CRepCase:=2:
    return rep:
end proc:

RegularGcdZeroDim := proc (R, Mvar, InInRep1, InInRep2, InTriSet)
    local maxpartialdeg, mymsg, N, M, dVsz, cVsz, rec1, rec2, InRep, NewRep,
    NewTriSet, i, rec3, TriSetDGS, ESTIDGS, polydegVec, polycoeVec, rec,
    tsdegsVec, tscoesVec, Nvec, NoOfPairs, polydegVecs, polycoeVecs,
    tsdegsVecs, tscoesVecs, result, themainvar, newR, poly, ts, InRep1,
    InRep2, NewRep1, NewRep2, R2: 
    N:=getIndexOfVar(R:-VarList, Mvar):
    M:=N-1:
    InRep1 := PartialDegsAssert (InInRep1):
    InRep2 := PartialDegsAssert (InInRep2):
    NewRep1:=CRepAssert(InRep1, 1):
    NewRep2:=CRepAssert(InRep2, 1):
    NewTriSet:=[seq(0, i = 1 .. M)]:
    for i from 1 to M do
        NewTriSet[i] := InTriSet[i]:
    end do:
    rec3:=MergeTriSetRep(M, NewTriSet):
    R2:=PolynomialRing(R:-prime, (R:-VarList)[2..-1]):
    TriSetDGS:=getTriSetDegs(R2, M, M, NewTriSet):

    ESTIDGS := Array(1 .. M+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := TriSetDGS[i-1]: end do:

    maxpartialdeg:=1:
    for i from 2 to M+1 do
       maxpartialdeg:=maxpartialdeg*(ESTIDGS[i]+1):
    end do:

    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, M+1):
    #dVsz:= 2*dVsz;
    dVsz:=dVsz*dVsz;
    #dVsz:=dVsz*(maxpartialdeg+1);
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    polydegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, M+1) + 1:
    cVsz:=cVsz*cVsz;
    #cVsz:=cVsz*(maxpartialdeg+1);
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    polycoeVec:=Array(1..cVsz, 'datatype'='integer[4]'):

    rec:=estimateThe2VecSiz_TriSet(R, TriSetDGS, NewTriSet):
    #rec1:=rec[1]*rec[1]:
    #rec2:=rec[2]*rec[2]:
    rec1:=rec[1]*(maxpartialdeg+1):
    rec2:=rec[2]*(maxpartialdeg+1):
    if((rec1+rec2+dVsz) > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    tsdegsVec:=Array(1..rec1, 'datatype'='integer[4]'):
    tscoesVec:=Array(1..rec2, 'datatype'='integer[4]'):
    Nvec := Array(1 .. dVsz, 'datatype'='integer[4]'):

    NoOfPairs:=ConnectorModuleNew1:-RegularGcdZeroDimCN(Nvec, polydegVec,
        polycoeVec, tsdegsVec, tscoesVec,N, M, NewRep1:-DGS,
        NewRep1:-CRep:-COEF, NewRep2:-DGS, NewRep2:-CRep:-COEF,
        TriSetDGS, rec3[1], rec3[2], rec3[3],  R:-prime):

    mymsg:=NoOfPairs:
    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    
    if NoOfPairs=0 then
        polydegVecs:=getAListOfVectors (polydegVec, 1, 1):
        polycoeVecs:=getAListOfVectors (polycoeVec, 1, 1):
        poly := Vec2Rep(R, polydegVecs[1], polycoeVecs[1]):
        return [[],poly];
    end if: 
    polydegVecs:=getAListOfVectors (polydegVec, NoOfPairs, 1):
    polycoeVecs:=getAListOfVectors (polycoeVec, NoOfPairs, 1):
    tsdegsVecs:=getAListOfVectors (tsdegsVec, NoOfPairs, M):
    tscoesVecs:=getAListOfVectors (tscoesVec, NoOfPairs, M):
    result:=[]:

    for i from 1 to NoOfPairs do
        #themainvar:=RecdenMainVariable(InRep1:-RecdenPoly);
        newR:=PolynomialRing(R:-prime, (R:-VarList)[1..Nvec[i]]):
        poly := Vec2Rep(newR, polydegVecs[i], polycoeVecs[i]):
        ts:=splitTriSetRep(R2, M, tsdegsVecs[i], tscoesVecs[i], R:-prime):
        result:=[op(result), [poly, ts]]:
    end do:
    return result:
end proc:


# assumption: InRep and InTriSet has the same main variable,
# which is Mvar.
IsInvertibleZeroDim := proc (R, Mvar, InInRep, InTriSet)
    local maxpartialdeg, mymsg, N, M, dVsz, cVsz, rec1, rec2, InRep, NewRep,
    NewTriSet, i, rec3, TriSetDGS, ESTIDGS, polydegVec, polycoeVec, rec,
    tsdegsVec, tscoesVec, Nvec, NoOfPairs, polydegVecs, polycoeVecs,
    tsdegsVecs, tscoesVecs, result, themainvar, newR, poly, ts:

    N:=nops(R:-VarList):
    M:=getIndexOfVar(R:-VarList, Mvar):
    InRep := PartialDegsAssert (InInRep):
    NewRep:=CRepAssert(InRep, 1):
    NewTriSet:=[seq(0, i = 1 .. M)]:
    for i from 1 to M do
        NewTriSet[i] := InTriSet[i]:
    end do:
    rec3:=MergeTriSetRep(M, NewTriSet):
    TriSetDGS:=getTriSetDegs(R, M, N, NewTriSet):
    
    ESTIDGS := Array(1 .. M+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := TriSetDGS[i-1]: end do:
    
    maxpartialdeg:=1:
    for i from 2 to M+1 do
       maxpartialdeg:=maxpartialdeg*(ESTIDGS[i]+1):
    end do:
    
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, M+1):
    #dVsz:=2*dVsz;
    dVsz:=dVsz*dVsz;
    #dVsz:=dVsz*(maxpartialdeg+1);
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    polydegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, M+1) + 1:
    cVsz:=cVsz*cVsz;
    #cVsz:=cVsz*(maxpartialdeg+1);
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    polycoeVec:=Array(1..cVsz, 'datatype'='integer[4]'):
    
    rec:=estimateThe2VecSiz_TriSet(R, TriSetDGS, NewTriSet):
    #rec1:=rec[1]*rec[1]:
    #rec2:=rec[2]*rec[2]:
    rec1:=rec[1]*(maxpartialdeg+1):
    rec2:=rec[2]*(maxpartialdeg+1):
    if((rec1+rec2+dVsz) > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    tsdegsVec:=Array(1..rec1, 'datatype'='integer[4]'):
    tscoesVec:=Array(1..rec2, 'datatype'='integer[4]'):
    Nvec := Array(1 .. dVsz, 'datatype' = 'integer[4]'):

    NoOfPairs:=ConnectorModuleNew1:-IsInvertibleZeroDimCN (Nvec, polydegVec,
        polycoeVec, tsdegsVec, tscoesVec, M, NewRep:-DGS, NewRep:-CRep:-COEF,
        TriSetDGS, rec3[1], rec3[2], rec3[3],  R:-prime):

    mymsg:=NoOfPairs:

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    polydegVecs:=getAListOfVectors (polydegVec, NoOfPairs, 1):
    polycoeVecs:=getAListOfVectors (polycoeVec, NoOfPairs, 1):
    tsdegsVecs:=getAListOfVectors (tsdegsVec, NoOfPairs, M):
    tscoesVecs:=getAListOfVectors (tscoesVec, NoOfPairs, M):
    result:=[]:
    for i from 1 to NoOfPairs do
        themainvar:=RecdenMainVariable(InRep:-RecdenPoly);
        newR:=PolynomialRing(R:-prime, (R:-VarList)[1..Nvec[i]]):
        poly := Vec2Rep(newR, polydegVecs[i], polycoeVecs[i]):
        ts:=splitTriSetRep(R, M, tsdegsVecs[i], tscoesVecs[i], R:-prime):
        result:=[op(result), [poly, ts]]:
    end do:
    return result:
end proc:


# Mvar for InInRep.
NormalForm :=proc(R, Mvar, InInRep, InTriSet, opt)
    local mymsg, M, N, NewRep, NewTriSet, TriSetDGS, i, rec3, res, InRep,
    ESTIDGS, dVsz, pdegVec, cVsz, coefVec, rep;

    N:=nops(R:-VarList):
    M:=getIndexOfVar(R:-VarList, Mvar):
    InRep := PartialDegsAssert (InInRep):
    NewRep:=CRepAssert(InRep, 1):
    NewTriSet:=[seq(0, i = 1 .. M)]:
    for i from 1 to M do
        NewTriSet[i] := InTriSet[i]:
    end do:
    rec3:=MergeTriSetRep(M, NewTriSet):
    TriSetDGS:=getTriSetDegs(R, M, N, NewTriSet):
    
    ESTIDGS := Array(1 .. M+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := TriSetDGS[i-1]: end do:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, M+1):
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, M+1) + 1:
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):
    
    mymsg:=ConnectorModuleNew1:-NormalFormCN(dVsz, pdegVec, cVsz, coefVec, M,
        NewRep:-DGS, NewRep:-CRep:-COEF, TriSetDGS, rec3[1], rec3[2], rec3[3],
        R:-prime, opt):

    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or  mymsg=-110)  then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(R:-prime, pdegVec, coefVec, R:-VarList):
    rep:-DGS:=0:
    rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF:=0:
    rep:-CRep:-PDGSV:=pdegVec:
    rep:-CRep:-COEFV:=coefVec:
    rep:-CRepCase:=2:
    return rep:
end proc:

IteratedResultantZeroDim :=proc(R, Mvar, InRep, InTriSet)
    local res, N, M, Rep, NewRep, rec3, TriSetDGS;
    N:=nops(R:-VarList):
    M:=getIndexOfVar(R:-VarList, Mvar):
    Rep := PartialDegsAssert (InRep):
    NewRep:=CRepAssert(Rep, 1):
    rec3:=MergeTriSetRep(N, InTriSet):
    TriSetDGS:=getTriSetDegs(R, N, N, InTriSet):
    
    #print("going into IterResultantZeroDimCN()"):
    res:=ConnectorModuleNew1:-IterResultantZeroDimCN(M, NewRep:-DGS,
        NewRep:-CRep:-COEF, N, TriSetDGS, rec3[1], rec3[2], rec3[3], R:-prime):
    #print("after IterResultantZeroDimCN()"):
    return res:
end proc:

ReduceCoeffs :=proc(R, InInRep, InTriSet)
    local mymsg, M, N, NewRep, TriSetDGS, i, rec3, res, InRep, ESTIDGS, dVsz,
    pdegVec, cVsz, coefVec, rep;

    N:=nops(R:-VarList):
    M:=N-1:
    InRep:=PartialDegsAssert(InInRep):
    NewRep:=CRepAssert(InRep, 1):
    rec3:=MergeTriSetRep(M, InTriSet):
    TriSetDGS:=getTriSetDegs(R, M, N, InTriSet):
    
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := TriSetDGS[i-1]: end do:
    ESTIDGS[N+1]:=InRep:-DGS[N]:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):

    mymsg:=ConnectorModuleNew1:-ReduceCoefficientCN(dVsz, pdegVec, cVsz,
        coefVec, N, NewRep:-DGS, NewRep:-CRep:-COEF, TriSetDGS, rec3[1],
        rec3[2], rec3[3], R:-prime):
    
    if mymsg=-10  then error  ERR_INTERRUPT: end if:
    if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
    if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
    
    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
    rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(R:-prime, pdegVec, coefVec, R:-VarList):
    rep:-DGS:=0:
    rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF:=0:
    rep:-CRep:-PDGSV:=pdegVec:
    rep:-CRep:-COEFV:=coefVec:
    rep:-CRepCase:=2:
    return rep:
end proc:

QuoModTriSet :=proc(R, InInRep1, InInRep2, InTriSet)
    local M, N, NewRep1, NewRep2, TriSetDGS, i, rec3, res, InRep1, InRep2,
    ESTIDGS, dVsz, pdegVec, cVsz, coefVec, rep;

    N:=nops(R:-VarList):
    InRep1:=PartialDegsAssert(InInRep1):
    InRep2:=PartialDegsAssert(InInRep2):
    if((InRep1:-DGS)[N-1] < (InRep2:-DGS)[N-1]) then
         return PolynomialConvertIn(R, 0):
    end if:
    M:=N-1:
    NewRep1:=CRepAssert(InRep1, 1):
    NewRep2:=CRepAssert(InRep2, 1):
    rec3:=MergeTriSetRep(M, InTriSet):
    TriSetDGS:=getTriSetDegs(R, M, N, InTriSet):
    ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
    for i from 2 to M+1 do ESTIDGS[i] := TriSetDGS[i-1]: end do:
    ESTIDGS[N+1]:=NewRep1:-DGS[N] - NewRep2:-DGS[N]:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
    if(dVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    pdegVec:=Array(1..dVsz, 'datatype'='integer[4]'):
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
    if(cVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
    coefVec:=Array(1..cVsz, 'datatype'='integer[4]'):

    ConnectorModuleNew1:-QuoModTriSetCN(dVsz, pdegVec, cVsz, coefVec, N,
        NewRep1:-DGS, NewRep1:-CRep:-COEF, NewRep2:-DGS, NewRep2:-CRep:-COEF,
        TriSetDGS, rec3[1], rec3[2], rec3[3], R:-prime):

    rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):

    rep:-RecdenPoly:=
        RecdenConnector:-CData2RecdenPoly(R:-prime, pdegVec, coefVec, R:-VarList):

    rep:-DGS:=0:
    rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
    rep:-CRep:-COEF:=0:
    rep:-CRep:-PDGSV:=pdegVec:
    rep:-CRep:-COEFV:=coefVec:
    rep:-CRepCase:=2:
    return rep:
end proc:

estimateForOnePoly:=proc(N, TriSetDgs)
    local ESTIDGS, i, dVsz, cVsz;
    ESTIDGS:=Array(1..N+1, 'datatype' = 'integer[4]'):
    for i from 1 to N do
      ESTIDGS[i+1]:=TriSetDgs[i]:
    end do:
    dVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1)+1:
    cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
    return [dVsz, cVsz]:
end proc:


estimateThe2VecSiz_TriSet:=proc(R, InTriSetDegs, InTriSet)
    local N, pdegSiz, coefSiz, i, rec;
    N:=ArrayNumElems(InTriSetDegs):
    ### N:=nops(InTriSet):
    pdegSiz:=0:
    coefSiz:=0:
    for i from 1 to N do
        rec:=estimateForOnePoly(i, InTriSetDegs):
        pdegSiz:=pdegSiz+rec[1]:
        coefSiz:=coefSiz+rec[2]:
    end do:
    return [pdegSiz, coefSiz]:
end proc:

Normalize:=proc(R, InTriSet)
   local mymsg, N, InTriSetDegs, rec, outPDGVECS, outCOEFVECS, bool, outTriSet;
   N := nops(R:-VarList):
   InTriSetDegs:=getTriSetDegs(R, N, N, InTriSet):

   rec:=estimateThe2VecSiz_TriSet(R, InTriSetDegs, InTriSet):
   if((rec[1]+rec[2]) > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   outPDGVECS:=Array(1..rec[1], 'datatype'='integer[4]'):
   outCOEFVECS:=Array(1..rec[2], 'datatype'='integer[4]'):
   rec:=MergeTriSetRep(N, InTriSet):
   bool:=ConnectorModuleNew1:-NormalizeTriSetCN(outPDGVECS, outCOEFVECS, N, InTriSetDegs, rec[1], rec[2], rec[3], R:-prime):

   mymsg:=bool;

   if mymsg=-10  then error  ERR_INTERRUPT: end if:
   if (mymsg=-100 or mymsg=-110)  then error  ERR_PRIME: end if:
   if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
   if(bool=-1) then return []: end if:
   outTriSet:=splitTriSetRep(R, N, outPDGVECS, outCOEFVECS, R:-prime):
   return outTriSet:
end proc:

PlainPrem_rc := proc (A_rc, B_rc, var, prime1)
  local AINrep, BINrep, RINrep;
  AINrep:=modpn:-PolynomialConvertInUNI(prime1, A_rc, var):
  BINrep:=modpn:-PolynomialConvertInUNI(prime1, B_rc, var):
  AINrep:=modpn:-CRepAssertUNI(AINrep):
  BINrep:=modpn:-CRepAssertUNI(BINrep):
  RINrep:=modpn:-PlainPseudoRemainder(AINrep, BINrep, var, prime1):
  return modpn:-PolynomialConvertOutUNI(RINrep):
end proc:


PlainPrem_rc_rec := proc (A_rc, B_rc, var, prime1)
  local AINrep, BINrep, RINrep;
  AINrep:=modpn:-PolynomialConvertInUNI(prime1, A_rc, var):
  BINrep:=modpn:-PolynomialConvertInUNI(prime1, B_rc, var):
  AINrep:=modpn:-CRepAssertUNI(AINrep):
  BINrep:=modpn:-CRepAssertUNI(BINrep):
  RINrep:=modpn:-PlainPseudoRemainder(AINrep, BINrep, var, prime1):
  return RINrep:-RecdenPoly:
end proc:

FastMonicDivUNI_rc := proc(A_rc, B_rc, var, prime1)
  local AINrep, BINrep, rec;
  AINrep:=modpn:-PolynomialConvertInUNI(prime1, A_rc, var):
  BINrep:=modpn:-PolynomialConvertInUNI(prime1, B_rc, var):
  AINrep:=modpn:-CRepAssertUNI(AINrep):
  BINrep:=modpn:-CRepAssertUNI(BINrep):
  rec:=modpn:-FastMonicDivUNI(AINrep, BINrep, var, prime1):
  return modpn:-PolynomialConvertOutUNI(rec:-Quo):
end proc:

#------------------------------------------------------------------------------
# Function:
#    {MultivariateResultant}
# Briefly:
#    {Multivariate polynomial resultant based fast interpolation.}
# Calling sequence:
#    {MultivariateResultant(R, ININrep1, ININrep2, case, p)}
# Input:
#    { R : a modpn polynomial ring }
#    { INrep1 : a modpn polynomial }
#    { INrep2 : a modpn polynomial }
#    { case : type of C representation for the output }
# Output:
#    { The resultant of `INrep1` and 'INrep2` }
#------------------------------------------------------------------------------
MultivariateResultant := proc (R, ININrep1, ININrep2,  p)
   local INrep1, INrep2, N , dgsr, rep1, rep2, pvsz, pdegVec, rbufsz, cvsz,
   coefVec, ESTIDGS, pVsz, cVsz, Nnew, mymsg, newinx, rep, i;
   #to assert two input dgs have the same variables.
   INrep1:=PartialDegsAssert(ININrep1):
   INrep2:=PartialDegsAssert(ININrep2):
   rep1:=CRepAssert(INrep1, 1):
   rep2:=CRepAssert(INrep2, 1):
   N:=ArrayNumElems(INrep1:-DGS):
   ESTIDGS := Array(1 .. N+1, 'datatype' = 'integer[4]'):
   for i from 2 to N+1 do
     ESTIDGS[i]:=(rep1:-DGS)[i-1]*(rep2:-DGS[N]) + (rep2:-DGS)[i-1]*(rep1:-DGS)[N] +1:
   end do:
  
   pVsz:=estimatePartialDegVecSizefrom2(ESTIDGS, N+1):
   if(pVsz > MEMORY_BOUND)  then error ERR_MEMORY: end if:
   pdegVec:=Array(1..pVsz, 'datatype'='integer[4]'):
   cVsz:=calSizDgsfrom2(ESTIDGS, N+1) + 1:
   if(cVsz>MEMORY_BOUND)  then error ERR_MEMORY: end if:
   coefVec:=Array(1..cVsz, 'datatype' = 'integer[4]'):

   Nnew:=Array(1..1, 'datatype'='integer[4]'):
   mymsg := ConnectorModuleNew1:-ResultantMultiCN(Nnew, pdegVec, coefVec,
         N, rep1:-DGS, rep1:-CRep:-COEF, rep2:-DGS, rep2:-CRep:-COEF, p):

   if mymsg=-10  then error  ERR_INTERRUPT: end if:
   if (mymsg=-100 or  mymsg=-110)  then error  ERR_PRIME: end if:
   if(mymsg = -200) then error ERR_FOURIERPRIME: end if:
   newinx:=N-Nnew[1]+1:
   rep:=Record('DGS', 'RecdenPoly', 'CRep', 'CRepCase'):
   rep:-RecdenPoly:=RecdenConnector:-CData2RecdenPoly(R:-prime, pdegVec, coefVec, (R:-VarList)[newinx..N]):
   rep:-DGS:=0:
   rep:-CRep:=Record('COEF', 'PDGSV', 'COEFV'):
   rep:-CRep:-COEF:=0:
   rep:-CRep:-PDGSV:=pdegVec:
   rep:-CRep:-COEFV:=coefVec:
   rep:-CRepCase:=2:
   return rep:
end proc;

end module:
#------------------------------------------------------------------------------
# Function:
# { }
# Briefly:
# { }
# Calling sequence:
# { }
# Input:
# { }
# Output:
# { }
#------------------------------------------------------------------------------

# uncomment for local
# comment for Maplesoft
#savelib('modpn');
