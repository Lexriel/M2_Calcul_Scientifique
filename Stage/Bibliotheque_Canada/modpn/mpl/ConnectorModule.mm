ConnectorModule := module ()  export TestMDag2CDag, TestRecden2C, TestC2Recden, MTFTFFT, MTFTFFTC, TFTFFTUNI, FASTDIVUNI, PLAINDIVUNI, FASTGCDUNI, PLAINGCDUNI, SUBPTREECRE, FastEval, FastInterp; local trampoline; option package;  

  trampoline := proc() local oldargs;
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


  TestMDag2CDag := trampoline('TestMDag2CDag', 
                'i'::(integer[4]), 
                'a'::(ARRAY(1 .. 'i', 1 .. 3, 'integer[4]')), 
                'LIB' = "modpn"); 


  TestRecden2C := trampoline('TestRecden2C', 
                'i'::(integer[4]), 
                'j'::(integer[4]), 
                'a'::(ARRAY(1 .. 'i', 'integer[4]')), 
                'b'::(ARRAY(1 .. 'j', 'integer[4]')), 
                'LIB' = "modpn"); 


  TestC2Recden := trampoline('TestC2Recden', 
                'i'::(integer[4]), 
                'pdV'::(ARRAY(1 .. 'i', 'integer[4]')), 
                'j'::(integer[4]), 
                'cV'::(ARRAY(1 .. 'j', 'integer[4]')), 
                 'LIB' = "modpn"); 


  MTFTFFT := trampoline('MulPolyTFTFFTCN', 
                 'N'::(integer[4]), 'rdgs'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
                 'rbsz'::(integer[4]), 
                 'rbuf'::(ARRAY(1 .. 'rbsz', 'integer[4]')), 
                 'p1dgs'::(ARRAY(1 .. 'N'+1, 'integer[4]')),
                 'p1bsz'::(integer[4]), 
                 'p1buf'::(ARRAY(1 .. 'p1bsz', 'integer[4]')), 
                 'p2dgs'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
                 'p2bsz'::(integer[4]), 
                 'p2buf'::(ARRAY(1 .. 'p2bsz', 'integer[4]')), 
                 'pvsz'::(integer[4]), 
                 'pv'::(ARRAY(1 .. 'pvsz', 'integer[4]')), 
                 'cvsz'::(integer[4]), 
                 'cv'::(ARRAY(1 .. 'cvsz', 'integer[4]')), 
                 'p'::(integer[4]), 
                 'LIB' = "modpn");


  MTFTFFTC := trampoline('MulPolyTFTFFTCNC', 
                 'N'::(integer[4]), 
                 'dgs1'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
                 'p1dgsSz'::(integer[4]), 
                 'p1dgs'::(ARRAY(1 .. 'p1dgsSz'+1, 'integer[4]')), 
                 'p1bsz'::(integer[4]), 
                 'p1buf'::(ARRAY(1 .. 'p1bsz', 'integer[4]')), 
                 'dgs2'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
                 'p2dgsSz'::(integer[4]), 
                 'p2dgs'::(ARRAY(1 .. 'p2dgsSz'+1, 'integer[4]')), 
                 'p2bsz'::(integer[4]), 
                 'p2buf'::(ARRAY(1 .. 'p2bsz', 'integer[4]')), 
                 'rdgs'::(ARRAY(1 .. 'N'+1, 'integer[4]')), 
                 'pvsz'::(integer[4]), 
                 'pv'::(ARRAY(1 .. 'pvsz', 'integer[4]')), 
                 'cvsz'::(integer[4]), 
                 'cv'::(ARRAY(1 .. 'cvsz', 'integer[4]')), 
                 'p'::(integer[4]), 
                  'LIB' = "modpn");


  TFTFFTUNI := trampoline('TFTFFTUNIC', 
                  'rd'::(integer[4]), 
                  'res'::(ARRAY(1 .. 'rd'+1, 'integer[4]')), 
                  'd1'::(integer[4]), 
                  'v1'::(ARRAY(1 .. 'd1'+1, 'integer[4]')), 
                  'd2'::(integer[4]), 
                  'v2'::(ARRAY(1 .. 'd2'+1, 'integer[4]')), 
                  'p'::(integer[4]), 
                  'LIB' = "modpn");


  FASTDIVUNI := trampoline('FASTDIVC', 
                 'rd'::(integer[4]), 
                 'rv'::(ARRAY(1 .. 'rd'+1, 'integer[4]')), 
                 'dq'::(integer[4]), 
                 'qv'::(ARRAY(1 .. 'dq'+1, 'integer[4]')), 
                 'Ad'::(integer[4]), 
                 'Av'::(ARRAY(1 .. 'Ad'+1, 'integer[4]')), 
                 'Bd'::(integer[4]), 
                 'Bv'::(ARRAY(1 .. 'Bd'+1, 'integer[4]')), 
                 'p'::(integer[4]), 
                 'LIB' = "modpn");


  PLAINDIVUNI := trampoline('PLAINDIVC', 
                  'rd'::(integer[4]), 
                  'rv'::(ARRAY(1 .. 'rd'+1, 'integer[4]')), 
                  'dq'::(integer[4]), 
                  'qv'::(ARRAY(1 .. 'dq'+1, 'integer[4]')), 
                  'Ad'::(integer[4]), 
                  'Av'::(ARRAY(1 .. 'Ad'+1, 'integer[4]')), 
                  'Bd'::(integer[4]), 
                  'Bv'::(ARRAY(1 .. 'Bd'+1, 'integer[4]')), 
                  'p'::(integer[4]), 
                 'LIB' = "modpn");


  PLAINGCDUNI := trampoline('PLAINGCDUNIC', 
                  'ud'::(integer[4]), 
                  'uv'::(ARRAY(1 .. 'ud'+1, 'integer[4]')), 
                  'vd'::(integer[4]), 
                  'vv'::(ARRAY(1 .. 'vd'+1, 'integer[4]')), 
                  'gd'::(integer[4]), 
                  'gv'::(ARRAY(1 .. 'gd'+1, 'integer[4]')), 
                  'Ad'::(integer[4]), 
                  'Av'::(ARRAY(1 .. 'Ad'+1, 'integer[4]')), 
                  'Bd'::(integer[4]), 
                  'Bv'::(ARRAY(1 .. 'Bd'+1, 'integer[4]')), 
                  'p'::(integer[4]), 
                'LIB' = "modpn");


  FASTGCDUNI := trampoline('FASTGCDUNIC', 
                  'ud'::(integer[4]), 
                  'uv'::(ARRAY(1 .. 'ud'+1, 'integer[4]')), 
                  'vd'::(integer[4]), 
                  'vv'::(ARRAY(1 .. 'vd'+1, 'integer[4]')), 
                  'gd'::(integer[4]), 
                  'gv'::(ARRAY(1 .. 'gd'+1, 'integer[4]')), 
                  'Ad'::(integer[4]), 
                  'Av'::(ARRAY(1 .. 'Ad'+1, 'integer[4]')), 
                  'Bd'::(integer[4]), 
                  'Bv'::(ARRAY(1 .. 'Bd'+1, 'integer[4]')), 
                  'p'::(integer[4]), 
                 'LIB' = "modpn");


  SUBPTREECRE := trampoline('subProdTreeCreWrapC', 
                  'h'::(integer[4]), 
                  'levels'::(integer[4]), 
                  'W'::(ARRAY(1 .. 'levels'+1, 'integer[4]')), 
                  'NoNodes'::(ARRAY(1 .. 'levels'+1, 'integer[4]')), 
                  'Bases'::(ARRAY(1 .. 'levels'+1, 'integer[4]')), 
                  'totSZ'::(integer[4]), 
                  'data'::(ARRAY(1 .. 'totSZ', 'integer[4]')), 
                  'itemNo'::(integer[4]), 
                  'itemSz'::(integer[4]), 
                  'p'::(integer[4]), 
                 'LIB' = "modpn");


  FastEval := trampoline('FastEvalWrapC', 
                  'n'::(integer[4]), 
                  'EvalPts'::(ARRAY(1 .. 'n', 'integer[4]')), 
                  'df'::(integer[4]), 
                  'fPtr'::(ARRAY(1 .. 'df'+1, 'integer[4]')), 
                  'h'::(integer[4]), 
                  'W'::(ARRAY(1 .. 'h'+2, 'integer[4]')), 
                  'NoNodes'::(ARRAY(1 .. 'h'+2, 'integer[4]')), 
                  'Bases'::(ARRAY(1 .. 'totSZ', 'integer[4]')), 
                  'data'::(ARRAY(1 .. 'h'+2, 'integer[4]')), 
                  'p'::(integer[4]), 
                 'LIB' = "modpn");


  FastInterp := trampoline('FastInterpWrapC', 
                  'n'::(integer[4]), 
                  'InterpedPts'::(ARRAY(1 .. 'n', 'integer[4]')), 
                  'EvaluatingPts'::(ARRAY(1 .. 'n', 'integer[4]')), 
                  'EvaluatedPts'::(ARRAY(1 .. 'n', 'integer[4]')), 
                  'h'::(integer[4]), 
                  'W'::(ARRAY(1 .. 'h'+2, 'integer[4]')), 
                  'NoNodes'::(ARRAY(1 .. 'h'+2, 'integer[4]')), 
                  'Bases'::(ARRAY(1 .. 'totSZ', 'integer[4]')), 
                  'data'::(ARRAY(1 .. 'h'+2, 'integer[4]')), 
                  'p'::(integer[4]), 
                 'LIB' = "modpn")

 
end module;
