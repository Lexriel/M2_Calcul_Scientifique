RecdenConnector := module()

    # exported names.
    # the CData2RecdenPoly is not exact symmetric RecdenPoly2CData.
    # going to C, the data passed is the super dense encoding.
    # coming back to Maple, the data passed is the recursive
    # data (partialDegVec, coefVec).
    export PartialDegs, RecdenPoly2CData, CData2RecdenPoly, cRPoly, 
           RecdenPoly2C2V, C2RecdenPolyUni, RecdenPoly2CUni, RecdenContent, 
           RecdenSquareFreePart,  RecdenMainVariable, RecdenIsZero, 
           RecdenSmallerMainVariable, RecdenPolynomialExactDivision, 
           RecdenInitial, RecdenTail, RecdenIsOne, RecdenGcd, getVarList, 
           correctVarList, RecdenNormalize, RecdenGcdInputHasTheSameRing, 
           EnlargeRecdentoR, RecdenPrem, RecdenMakeConstant, 
           RecdenMakeTheSameRing, Recdendegrpoly, RecdenPremUni;

    # local names.
    local  ArrReverse, searchPartialDeg, Rec2CData, Rec2C, List2Vec, 
           CData2RecdenPolyInner, RecdenPoly2C2VData, ReverseArr, shiftArr, 
           estimatePartialDegVecSize;
    # options.
    option package;

    # fields:


#------------------------------------------------------------------------------
# Function:
#    {List2Vec }
# Briefly:
#    { Converting a Maple list to a Maple Vector}
# Calling sequence:
#    {List2Vec(n, L) }
# Input:
#    {n : The size of the input list. }
#    {L : The list. }
# Output:
#    { The vector.}
#------------------------------------------------------------------------------

    List2Vec := proc (n, L)
        local V, i;
        V := Array(1 .. n, 'datatype' = 'integer[4]');
        for i to n do 
           V[i] := L[i];
        end do;
        return V:
    end proc:


#------------------------------------------------------------------------------
# Function:
#    {correctVarList }
# Briefly:
#    { Inner function }
# Calling sequence:
#    { }
# Input:
#    { }
# Output:
#    { }
#------------------------------------------------------------------------------
    correctVarList :=proc(rp1)
      local degr, i, ii, m, n, newVarList, oldVarList;
       oldVarList:=op(rp1)[1][2]:
       n:=nops(oldVarList):
       ii:=1:
       for i from 1 to n do
          degr:=Recdendegrpoly(rp1, oldVarList[i]):
          if(degr <> 0) then  
             ii:=i;   
             break; 
          end if;
       end do:
       m := n-ii+1:
       #print("m:", m);
       newVarList:=[seq(0, i = 1 .. m)];
       for i from ii to n do
           newVarList := subsop(i-ii+1=oldVarList[i], newVarList):
       end do:
       return newVarList:
    end proc:


    getVarList := proc(rp1)
       return op(rp1)[1][2]:
    end proc:

#------------------------------------------------------------------------------
# Function:
#    { RecdenPoly2CUni }
# Briefly:
#    { Converting a univariate Recden polynomial into Modpn representation. }
# Calling sequence:
#    { RecdenPoly2CUni(recpoly). }
# Input:
#    { recpoly: a Recden Polynomial.}
# Output:
#    { A Modpn univariate polynomial representation.}
#------------------------------------------------------------------------------
    RecdenPoly2CUni := proc(recpoly)
        local V, n, opCoef;
        opCoef:=op(recpoly)[2]:
        n:=nops(opCoef):
        V:=List2Vec(n, opCoef):
        return [n-1, V];
    end proc:



#------------------------------------------------------------------------------
# Function:
#    {C2RecdenPolyUni}
# Briefly:
#    {Generate a univariate Maple polynomial from the input.}
# Calling sequence:
#    {C2RecdenPolyUni(p, V, var) }
# Input:
#    { p: the prime. }
#    { V: the coefficient vector. }
#    {var: the variable of the univariate polynomial.}
# Output:
#    { A univariate Maple polynomial.}
#------------------------------------------------------------------------------

    C2RecdenPolyUni := proc(p, V, var)
        local rop;
        rop:=[p, [var], []], convert(V, list):
        return POLYNOMIAL(rop):
    end proc;

    #===============================================#
    #        Multi-case                             #
    #===============================================#
    # procedure: Reversing the array.


#------------------------------------------------------------------------------
# Function:
#    {ArrReverse}
# Briefly:
#    {To reverse the coefficients order of the input Array. }
# Calling sequence:
#    {ArrReverse(n, Arr) }
# Input:
#    { n: the size of the input Array. }
#    { Arr: a input Array.}
# Output:
#    { The same input Array with coefficients order reversed.}
#------------------------------------------------------------------------------
    ArrReverse := proc (n, Arr)
        local i, tmp;
        for i to iquo(n, 2) do
            tmp := Arr[i]; 
            Arr[i] := Arr[n-i+1]; 
            Arr[n-i+1] := tmp;
        end do;
        return Arr;
    end proc;


#------------------------------------------------------------------------------
# Function:
#    { searchPartialDeg }
# Briefly:
#    { Inner function for PartialDegs(), the next procedure.}
# Calling sequence:
#    { }
# Input:
#    { }
# Output:
#    { }
#------------------------------------------------------------------------------
    searchPartialDeg := proc (level, Arr, coefs)
        local degr, fir, i, n;
        if level = 0 or type(coefs, integer) then 
           return Arr;
        end if;
        degr := nops(coefs)-1;
        if Arr[level] < degr then 
           Arr[level] := degr;
        end if;
        n := nops(coefs);
        for i to n do
            fir := coefs[i];
            searchPartialDeg(level-1, Arr, fir)
        end do;
        return Arr:
    end proc;

    # procedure: trace the rpoly's partial degrees.
#------------------------------------------------------------------------------
# Function:
#    {PartialDegs}
# Briefly:
#    {To compute the partial degreess of the input Recden polynomial 'renpoly'.}
# Calling sequence:
#    {PartialDegs(renpoly) }
# Input:
#    {renpoly: a Recden polynomial.}
# Output:
#    {A Record [varnos, Arr, rpop[2]], varnos is the number of variables for the input polynomial, Arr is the partial degrees vector, rpop[2] is the second entry in the Recden polynomial encoding.}
#------------------------------------------------------------------------------
    PartialDegs := proc (renpoly)
        local Arr, rpop, varList, varnos;
        rpop := op(renpoly);
        varList := rpop[1][2];
        varnos := nops(varList);
        Arr := Array(1 .. varnos, 'datatype' = 'integer[4]');
        searchPartialDeg(varnos, Arr, rpop[2]);
        return [varnos, Arr, rpop[2]]:
    end proc;

    # Inner function of function Rec2C.
    # INPUT: Arr is an empty buffer to keep the coef array forworded to C. base is the base to start fetch current coefficients.   accum is the size of coefficients at each level.
    Rec2CData := proc (Arr, base, accum, level, coefsdata)
        local ac, b, i, n;
        b := base;
        if level = 0 then 
            Arr[b] := coefsdata; 
            return Arr;
        end if;
        if type(coefsdata, integer) then
            Arr[b] := coefsdata; 
            return Arr;
        end if;
        n := nops(coefsdata);
        for i to n do
            ac := accum[level];
            Rec2CData(Arr, b, accum, level-1, coefsdata[i]);
            b := b+ac;
        end do;
        return Arr:
    end proc;

    # Inner function of funciton RecPoly2CData.
    # INPUT: n is the size of degs array, or say the number of variables.
    #        dgs is the partial degrees.
    #        coefsdata is the rpoly's coefficient data.
    # OUTPUT: the Array which going to be passed to C.

    Rec2C := proc (n, dgs, coefsdata)
        local MArr, accum, i, size;
        size := 1; 
        accum := Array(1 .. n, 'datatype' = 'integer[4]');
        accum[1] := 1;
        for i from 2 to n do
            accum[i] := accum[i-1]*(dgs[i-1]+1);
        end do;
        for i to n do
            size := size*(dgs[i]+1);
        end do;
        MArr := Array(1 .. size, 'datatype' = 'integer[4]');
        Rec2CData(MArr, 1, accum, n, coefsdata);
        return MArr:
    end proc;

    # Converting RecDen represenatin into the C level coefficient array layout.
    #  INPUT: rpoly, the RecurDen polynomial.
    #   OUTPUT: The C coefficient array.
#------------------------------------------------------------------------------
# Function:
#    {RecdenPoly2CData}
# Briefly:
#    {To convert the input Recden polynomial into the conresponding C-Cube coefficient vector. }
# Calling sequence:
#    { RecdenPoly2CData(renpoly). }
# Input:
#    { renpoly: a Recden polynomial.}
# Output:
#    { A Maple integer[4] Vector keeps the coefficients of the input Recden polynomial
#      in the C-Cube polynomial encoding's layout.}
#------------------------------------------------------------------------------
    RecdenPoly2CData := proc (renpoly)
        local MArr, rec;
        rec := PartialDegs(renpoly);
        MArr := Rec2C(rec[1], rec[2], rec[3]);
        return MArr:
    end proc;




#------------------------------------------------------------------------------
# Function:
#    { cRPoly }
# Briefly:
#    { Inner function of CData2RecdenPolyInner}
# Calling sequence:
#    { }
# Input:
#    { }
# Output:
#    { }
#------------------------------------------------------------------------------
    cRPoly := proc (level, pdegV, coefV)
        local  d, i, l, Arr;

        if pdegV[1] <= 0 then return 0 end if;

        d := pdegV[pdegV[1]]; pdegV[1] := pdegV[1]-1;

        if d = -1 then
            l := 0
        else
            if level = 1 then
                Arr := Array(1 .. d + 1, 'datatype' = 'integer[4]');
                for i from 1 to d+1 do
                    coefV[1]:=coefV[1]+1:
                    Arr[i]:=coefV[coefV[1]]:
                end do:
                l:=convert(Arr, 'list'):
            else
                l:=[]:
                for i from d+1 by -1 to 1 do
                    l:=[cRPoly(level-1, pdegV, coefV), op(l)]:
                end do:
            end if
        end if;
        return l:
    end proc;



    # procedure: constructing a Recurden poly from 2 C vectors
    #(compact partial degree Vector,  compact coefficient Vector.)






#------------------------------------------------------------------------------
# Function:
#    { cRPoly }
# Briefly:
#    { Inner function of CData2RecdenPoly.}
# Calling sequence:
#    { }
# Input:
#    { }
# Output:
#    { }
#------------------------------------------------------------------------------
    CData2RecdenPolyInner := proc (p, pdegV, coefV, varList)
        local  a, b, r, rop, level;
        a := pdegV[1]; b := coefV[1];
         pdegV[1] := pdegV[1]+1;
        coefV[1] := 1;
        #x := stp;
        level:=nops(varList):
        r := cRPoly(level, pdegV, coefV);
        #print("op(r)=", op(r));
        rop := [p, varList, []], r;
        pdegV[1] := a: coefV[1] := b:
        return POLYNOMIAL(rop):
    end proc;



#------------------------------------------------------------------------------
# Function:
#    { CData2RecdenPoly }
# Briefly:
#    { To convert an input C-Cube polynomial into a Maple DAG polynomial.}
# Calling sequence:
#    { CData2RecdenPoly(p, pdegV, coefV, varList). }
# Input:
#    { p: the prime number.}
#    { pdegV: The partial degree vector.}
#    { coefV: The coefficient vector.}
#    { varList: the variable list.}
# Output:
#    { A Maple polynomial.}
#------------------------------------------------------------------------------
    CData2RecdenPoly := proc(p, pdegV, coefV, varList)
         local  a, b, c, d, dif, i, r, rop, rp1, vList;

#         print("pdegV =", pdegV):
#         print("coefV =", coefV):
#         print("varList =", varList):

         a := pdegV[1]; b := coefV[1];

         if a =1 and b =1 then
           c := pdegV[2]; d := coefV[2];
           if (c=0) then
              rop := [p, [], []], d;
              return POLYNOMIAL(rop):
           end if:
         end if:

         rp1:=CData2RecdenPolyInner(p, pdegV, coefV, varList):
         vList:=correctVarList(rp1):
         #print("vList", vList):
         #print("rp1=", op(rp1));
         r:=op(rp1)[2]:
         dif:=nops(varList)-nops(vList):
         for i from 1 to dif do
               r:=r[1];
         end do:
         rop:=[p, vList, []], r;
         return POLYNOMIAL(rop):
    end proc;


    # Developer level function.
    RecdenNormalize:= proc(rp1)
         local dif, i, p, r, rop, vList, varList;
         p:=op(rp1)[1][1]:
         varList:=op(rp1)[1][2]:
         vList:=correctVarList(rp1):
         r:=op(rp1)[2]:
         dif:=nops(varList)-nops(vList):
         for i from 1 to dif do
               r:=r[1];
         end do:
         rop:=[p, vList, []], r;
         return POLYNOMIAL(rop):
    end proc:


#------------------------------------------------------------------------------
# Function:
#    { RecdenInitial }
# Briefly:
#    { Returns the initial of the input Recden polynomial. }
# Calling sequence:
#    { RecdenInitial(rp1)}
# Input:
#    { A Recden Polynomial.}
# Output:
#    { The inital of the input Recden polynomial which is also a Recden polynomial.}
#------------------------------------------------------------------------------

   RecdenInitial:= proc (rp1)
       local R,A,M,X,E,Y;
       if Algebraic:-RecursiveDensePolynomials:-isconstant(rp1) then
          error("It's a constant."):
       end if;
       R,A := op(rp1);
       M,X,E := op(R);
       Y:=X[1];
       return Algebraic:-RecursiveDensePolynomials:-lcrpoly(rp1,Y);
    end proc;

    RecdenTail:= proc (rp1)
        return Algebraic:-RecursiveDensePolynomials:-redrpoly(rp1):
    end proc;


#------------------------------------------------------------------------------
# Function:
#    { RecdenMainVariable }
# Briefly:
#    { Returns the main variable of the input Recden polynomial. }
# Calling sequence:
#    { RecdenMainVariable(rp1)}
# Input:
#    { rp1: a Recden polynomial.}
# Output:
#    { The main variable of the input Recden polynomial.}
#------------------------------------------------------------------------------

    RecdenMainVariable := proc(rp1)
       local i, no;
       if Algebraic:-RecursiveDensePolynomials:-isconstant(rp1) then
          error("It's a constant."):
       else
          no:=nops(rp1[1][2]):
          for i from 1 by no do
             if Recdendegrpoly(rp1, ((op(rp1))[1][2])[i]) <> 0 then
                 return ((op(rp1))[1][2])[i]:
             end if:
          end do:
       end if:
    end proc:


    # degin starts from the second slot. and the final size
    # will be kept in the first slot.

#------------------------------------------------------------------------------
# Function:
#    { RecdenPoly2C2VData }
# Briefly:
#    { inner function of RecdenPoly2C2V(). }
# Calling sequence:
#    { }
# Input:
#    { }
# Output:
#    { }
#------------------------------------------------------------------------------
    RecdenPoly2C2VData := proc (degincofin, level, pdegV, coefV, Data)
        local i, n, rec;
        rec:=degincofin:
        if Data = 0 then
            pdegV[rec[1]]:=-1:
            return [rec[1]+1, rec[2]]:
        end if:

        n := nops(Data):
        if level = 1 then
            for i from 1 to n do
                coefV[rec[2]]:=Data[i]:
                rec[2]:=rec[2]+1:
            end do;
            pdegV[rec[1]]:=n-1:
            return [rec[1]+1, rec[2]]
        end if;

        pdegV[rec[1]]:=n-1:
        rec[1]:=rec[1]+1;
        for i from n by -1 to 1 do
            rec:=(RecdenPoly2C2VData(rec, level-1, pdegV, coefV, Data[i])):
        end do:

        return [rec[1], rec[2]]:
    end proc:

   # calSizDgs:=proc(dgs, n)
   #     sz:=1:
   #     for i from 1 to n do
   #         sz:=sz*(dgs[i]+1):
   #     end do:
   #     sz:
   # end proc:

#------------------------------------------------------------------------------
# Function:
#    { estimatePartialDegVecSize }
# Briefly:
#    { inner function of RecdenPoly2C2V(). }
# Calling sequence:
#    { }
# Input:
#    { }
# Output:
#    { }
#------------------------------------------------------------------------------
    estimatePartialDegVecSize := proc(dgs, n)
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

#------------------------------------------------------------------------------
# Function:
#    { ReverseArr }
# Briefly:
#    { inner function of RecdenPoly2C2V(). }
# Calling sequence:
#    { }
# Input:
#    { }
# Output:
#    { }
#------------------------------------------------------------------------------
    ReverseArr:=proc(A, n)
        local i, n2, tmp;
        n2:=iquo(n,2);
        for i from 1 to n2 do
            tmp:=A[i]:
            A[i]:=A[n-i+1]:
            A[n-i+1]:=tmp:
        end do:
        return A:
    end proc:


#------------------------------------------------------------------------------
# Function:
#    { shiftArr }
# Briefly:
#    { inner function of RecdenPoly2C2V(). }
# Calling sequence:
#    { }
# Input:
#    { }
# Output:
#    { }
#------------------------------------------------------------------------------
    shiftArr:=proc(A,n,m)
        local i;
        for i from 1+m to n do
            A[i-m]:=A[i]:
        end do:
        return A:
    end proc;



#------------------------------------------------------------------------------
# Function:
#    { RecdenPoly2C2V }
# Briefly:
#    { To convert the input Recden polynomial into 2-VECTOR representation. }
# Calling sequence:
#    { RecdenPoly2C2V(renpoly) }
# Input:
#    { renpoly: A Recden polynomial.}
# Output:
#    { Returns the 2-Vector represention of the input Recden polynomial.}
#------------------------------------------------------------------------------
    RecdenPoly2C2V := proc (renpoly)
        local coefV, cofsz, dgs, n, pdegR, pdegV, pdgsz, psz, rec, rpop, varList;
        rpop := op(renpoly):
        varList := rpop[1][2]:
        n := nops(varList):
        dgs := Array(1 .. n, 'datatype' = 'integer[4]');
        dgs:=searchPartialDeg (n, dgs, rpop[2]):
        cofsz:=calSizDgs(dgs, n)+1:
        coefV:= Array(1 .. cofsz, 'datatype' = 'integer[4]');
        pdgsz:=estimatePartialDegVecSize(dgs,n):
        pdegV:= Array(1 .. pdgsz, 'datatype' = 'integer[4]');
        rec:=RecdenPoly2C2VData([2,2], n, pdegV, coefV, rpop[2]):
        psz:=rec[1]-2:
        pdegV:=shiftArr(pdegV,psz+1,1):
        pdegV[psz+1]:=0:
        pdegR:=ReverseArr(pdegV, psz+1):
        pdegR[1]:=psz:
        pdegR:
        coefV[1]:=rec[2]-1:
        return [pdegR, coefV]:
    end proc;



#------------------------------------------------------------------------------
# Function:
#    { RecdenIsZero }
# Briefly:
#    { To test if the input Recden polynomial is zero or not}
# Calling sequence:
#    { RecdenIsZero(rp1)}
# Input:
#    { rp1: a Recden polynomial.}
# Output:
#    { If rp1 equals to zero then return true, otherwise returns false.}
#------------------------------------------------------------------------------
    RecdenIsZero  := proc(rp1)
        return evalb(op(2,rp1)=0)
    end proc;


#------------------------------------------------------------------------------
# Function:
#    { RecdenIsOne }
# Briefly:
#    { To test if the input Recden polynomial is one or not}
# Calling sequence:
#    { RecdenIsZero(rp1)}
# Input:
#    { rp1: a Recden polynomial.}
# Output:
#    { If rp1 equals to one then return true, otherwise returns false.}
#------------------------------------------------------------------------------
    RecdenIsOne     := proc(rp1)
        return evalb(op(2,rp1)=1)
    end proc;





#------------------------------------------------------------------------------
# Function:
#    { RecdenGcd }
# Briefly:
#    { Computer the GCD of two input Recden polynomials.}
# Calling sequence:
#    { RecdenGcd(rp1, rp2)}
# Input:
#    { rp1: a Recden polynomial.}
#    { rp2: a Recden polynomial.}
# Output:
#    { The GCD of two input Recden polynomials.}
#------------------------------------------------------------------------------
    RecdenGcd  := proc(rp1, rp2)
     local one, res, rp11, rp12;
     one := Algebraic:-RecursiveDensePolynomials:-rpoly(1,(op(rp1))[1][2]):
     one := Algebraic:-RecursiveDensePolynomials:-modrpoly( one, (op(rp1))[1][1] );
     if whattype(rp1)=integer or whattype(rp2)=integer then
        return one:
     end if:
     rp11:=RecdenNormalize(rp1):
     rp12:=RecdenNormalize(rp2):
     res:=Algebraic:-RecursiveDensePolynomials:- PGCD(rp11, rp12):
     if(whattype(res)=integer) then
       res := Algebraic:-RecursiveDensePolynomials:-rpoly(res,(op(rp1))[1][2]):
       res := Algebraic:-RecursiveDensePolynomials:-modrpoly(res, (op(rp1))[1][1] );
     end if:
     return res:
    end proc:



#------------------------------------------------------------------------------
# Function:
#    { RecdenGcd }
# Briefly:
#    { Computer the GCD of two input Recden polynomials. ASSUME
#      rp1, rp2 has the same recden ring.}
# Calling sequence:
#    { RecdenGcd(rp1, rp2)}
# Input:
#    { rp1: a Recden polynomial.}
#    { rp2: a Recden polynomial.}
# Output:
#    { The GCD of two input Recden polynomials.}
#------------------------------------------------------------------------------
    RecdenGcdInputHasTheSameRing  := proc(rp1, rp2)
     local one, res;
     one := Algebraic:-RecursiveDensePolynomials:-rpoly(1,(op(rp1))[1][2]):
     one := Algebraic:-RecursiveDensePolynomials:-modrpoly( one, (op(rp1))[1][1] );
     if whattype(rp1)=integer or whattype(rp2)=integer then
        return one:
     end if:
     #rp11:=RecdenNormalize(rp1):
     #rp12:=RecdenNormalize(rp2):
     #print("rp11", op(rp11));
     #print("rp12", op(rp12));
     res:=Algebraic:-RecursiveDensePolynomials:- PGCD(rp1, rp2):
     if(whattype(res)=integer) then
       res := Algebraic:-RecursiveDensePolynomials:-rpoly(res,(op(rp1))[1][2]):
       res := Algebraic:-RecursiveDensePolynomials:-modrpoly(res, (op(rp1))[1][1] );
     end if:
     return res:
    end proc:



#------------------------------------------------------------------------------
# Function:
#    { RecdenContent }
# Briefly:
#    { Compute the content of the input Recden polynomial.}
# Calling sequence:
#    { RecdenContent(rp1) }
# Input:
#    { rp1: a Recden polynomial.}
# Output:
#    { The content of the input Recden polynomial.}
#------------------------------------------------------------------------------
    RecdenContent := proc(rp1)
        local v, rp2;
        if Algebraic:-RecursiveDensePolynomials:-isconstant(rp1) then return (rp1) end if;
        if Algebraic:-RecursiveDensePolynomials:-isunivariate(rp1) then
            rp2 := Algebraic:-RecursiveDensePolynomials:-rpoly(1,(op(rp1))[1][2]):
            rp2 := Algebraic:-RecursiveDensePolynomials:-modrpoly( rp2, (op(rp1))[1][1] );
            return rp2:
        end if;
        v := (op(rp1))[1][2][1];
        return Algebraic:-RecursiveDensePolynomials:-contrpoly(rp1,v);
    end proc;



#------------------------------------------------------------------------------
# Function:
#    { RecdenPolynomialExactDivision }
# Briefly:
#    { A exact division of two Recden polynomials.}
# Calling sequence:
#    { RecdenPolynomialExactDivision(rp1, rp2)}
# Input:
#    { rp1: The divident Recden polynomial.}
#    { rp1: The dividor Recden polynomial.}
# Output:
#    { The quotient}
#------------------------------------------------------------------------------
    RecdenPolynomialExactDivision := proc (rp1, rp2)
        local bool, q;
        if Algebraic:-RecursiveDensePolynomials:-isconstant(rp2) then 
           return (rp1);
        end if;
        bool := Algebraic:-RecursiveDensePolynomials:-divrpoly(rp1, rp2, 'q');
        if (bool = false) then 
           error("Talk to me! "); 
        end if;
        return q;
    end proc;



#------------------------------------------------------------------------------
# Function:
#    { RecdenSquareFreePart }
# Briefly:
#    { To compute the square free part of the input Recden polynomial.}
# Calling sequence:
#    {  RecdenSquareFreePart(rp1) }
# Input:
#    { rp1: a Recden polynomial.}
# Output:
#    { The square-free part of the input Recden polynomial.}
#------------------------------------------------------------------------------
    RecdenSquareFreePart  := proc (rp1)
        local p,g,c,v,f,bool,res,sqfrc,sqfrf;
        if Algebraic:-RecursiveDensePolynomials:-isconstant(rp1) then 
           return (rp1);
        end if;
        p := (op(rp1))[1][1];
        if (p <> 0) and evalb(irem(Recdendegrpoly(rp1, RecdenMainVariable(rp1)), p)  = 0) then
           error("The characteristic divides the degree: not supported yet") end if;
        #v := (op(rp1))[1][2][1];
        c := RecdenContent(rp1);
        if Algebraic:-RecursiveDensePolynomials:-isconstant(c) then
           f := rp1;
           sqfrc := c;
        else
          sqfrc := RecdenSquareFreePart (c);
          bool := Algebraic:-RecursiveDensePolynomials:-divrpoly(rp1, c, f);
          if (bool = false) then error("Talk to me! "); end if;
        end if;
        if evalb(Recdendegrpoly(f, RecdenMainVariable(f)) = 1) then return (Algebraic:-RecursiveDensePolynomials:-mulrpoly(sqfrc,f)) end if;
        #g :=Algbraic:-RecursiveDensePolynomials:- PGCD(f, Algebraic:-RecursiveDensePolynomials:-diffrpoly(f));
        g:=RecdenGcd(f, Algebraic:-RecursiveDensePolynomials:-diffrpoly(f));
        if Algebraic:-RecursiveDensePolynomials:-isconstant(g) then
           res :=Algebraic:-RecursiveDensePolynomials:- mulrpoly(sqfrc,f);
        else
           bool := Algebraic:-RecursiveDensePolynomials:-divrpoly(f, g,  'sqfrf');
           if (bool = false) then 
              error("Talk to me! "); 
           end if;
           res :=Algebraic:-RecursiveDensePolynomials:- mulrpoly(sqfrc,sqfrf);
        end if;
        return res:
     end proc;



#------------------------------------------------------------------------------
# Function:
#    { RecdenSmallerMainVariable }
# Briefly:
#    { To check if the first input Recden polynomial has smaller main
#      variable comparing to the second one.}
# Calling sequence:
#    { RecdenSmallerMainVariable(rp1, rp2, R) }
# Input:
#    { rp1: the first input Recden polynomial.}
#    { rp2: the second input Recden polynomial.}
#    { R: a modpn ring.}
# Output:
#    { Returns ture if main variable of 'rp1' is smaller than the one of 'rp2'.}
#------------------------------------------------------------------------------

    RecdenSmallerMainVariable := proc (rp1, rp2, R)
       local i, v1, v2, i1, i2, bool1, bool2:

       bool1:=Algebraic:-RecursiveDensePolynomials:-isconstant(rp1):
       bool2:=Algebraic:-RecursiveDensePolynomials:-isconstant(rp2):
       if (bool1) and (bool2) then
          return false:
       end if:
       if (bool1) and (not bool2) then
          return true:
       end if:
       if (not bool1) and (bool2) then
          return false:
       end if:

       i1 := 0; i2 := 0;
       v1:=RecdenMainVariable(rp1);
       v2:=RecdenMainVariable(rp2);
       for i from 1 to nops(R:-VarList) do
            if v1=(R:-VarList)[i] then i1:=i; end if:
            if v2=(R:-VarList)[i] then i2:=i; end if:
       end do:
       if ((i1 = 0) or (i2 = 0)) then 
          error "Should not happen!!!"; 
       end if;
       return i1> i2;
    end proc;





#------------------------------------------------------------------------------
# Function:
#    { EnlargeRecdentoR }
# Briefly:
#    { Inner function of RecdenMakeTheSameRing. }
# Calling sequence:
#    { }
# Input:
#    { }
# Output:
#    { }
#------------------------------------------------------------------------------
    EnlargeRecdentoR := proc(VarList1, Prime1, rp1)
#            local coeffs, i, incr, newrop, rop, N, M, oldVarList, mp1, newrp1;
            local N, M, oldVarList, mp1, newrp1;
            N:=nops(VarList1):
            oldVarList:=op(rp1)[1][2]:
            M:=nops(oldVarList):
            if (N = M) then return rp1: end if:
            if (N < M) then print("smaller! WRONG"): end if:

            mp1:=Algebraic:-RecursiveDensePolynomials:-rpoly(rp1):
            newrp1:=Algebraic:-RecursiveDensePolynomials:-rpoly(mp1, VarList1):
            newrp1:= Algebraic:-RecursiveDensePolynomials:-modrpoly(newrp1, Prime1);
            return newrp1:
   end proc:



#------------------------------------------------------------------------------
# Function:
#    { RecdenMakeTheSameRing}
# Briefly:
#    { Make the two input Recden polynomials have the same modpn ring.}
# Calling sequence:
#    { RecdenMakeTheSameRing(rp1, rp2)}
# Input:
#    { rp1: A Recden polynomial.}
#    { rp2: A Recden polynomial.}
# Output:
#    { A list of two Recden polynomials. They are euqual to the two input Recden polynomials and guaranteed that the out polynomials have the same modpn ring.}
#------------------------------------------------------------------------------

   RecdenMakeTheSameRing := proc (rp1, rp2)
      local N1, N2;
      N1:=nops(op(rp1)[1][2]):
      N2:=nops(op(rp2)[1][2]):
      ASSERT((op(rp1)[1][2])[N1] =  (op(rp2)[1][2])[N2]):
      if N1>N2 then
          return [rp1, EnlargeRecdentoR(op(rp1)[1][2], op(rp1)[1][1], rp2)]:
      end if:
      return [EnlargeRecdentoR(op(rp2)[1][2], op(rp2)[1][1], rp1), rp2]:
   end proc:



#------------------------------------------------------------------------------
# Function:
#    { RecdenMakeConstant }
# Briefly:
#    { Generate a constant Recden polynomial.}
# Calling sequence:
#    { RecdenMakeConstant(R, con)}
# Input:
#    { R: the modpn Ring.}
#    { con: A number.}
# Output:
#    { A constant Recden polynomial.}
#------------------------------------------------------------------------------
   RecdenMakeConstant :=proc (R, con)
     local conrpoly;
     conrpoly := Algebraic:-RecursiveDensePolynomials:-rpoly(con, R:-VarList):
     conrpoly := Algebraic:-RecursiveDensePolynomials:-modrpoly(conrpoly, R:-prime);
     return conrpoly:
   end proc:


#------------------------------------------------------------------------------
# Function:
#    { RecdenPrem }
# Briefly:
#    { To compute the pseudo remainder.}
# Calling sequence:
#    { RecdenPrem(rp1, rp2) }
# Input:
#    { rp1: A recden polynomial.}
#    { rp2: A recden polynomial.}
# Output:
#    { Pseduo remainder of 'rp1' is divided by 'rp2'.}
#------------------------------------------------------------------------------
   RecdenPrem := proc (rp1, rp2)
      local newrp1, newrp2, newrps:
      newrps:=RecdenMakeTheSameRing(rp1, rp2):
      newrp1:=newrps[1]:
      newrp2:=newrps[2]:
      return Algebraic:-RecursiveDensePolynomials:-premrpoly(newrp1, newrp2);
   end proc:


#------------------------------------------------------------------------------
# Function:
#    { RecdenPremUni }
# Briefly:
#    { To compute the pseudo remainder of two univariate Recden polynomials.}
# Calling sequence:
#    { RecdenPremUni(rp1, rp2)}
# Input:
#    { rp1: a unvariate Recden polynomial.}
#    { rp2: a unvariate Recden polynomial.}
# Output:
#    { The pesudo remainder of rp1 divided by rp2.}
#------------------------------------------------------------------------------
   RecdenPremUni := proc (rp1, rp2)
      local newrp1, newrp2, newrps, df, mp1, mp2, d1, d2, var, res:
      mp1:=Algebraic:-RecursiveDensePolynomials:-rpoly(rp1):
      mp2:=Algebraic:-RecursiveDensePolynomials:-rpoly(rp2):
      d1:=degree(mp1):
      d2:=degree(mp2):
      if(d1=0) then return rp1: end if:
      df:=d1-d2:
      if(df<0) then
        return rp1:
      end if:
      var:=op(rp1)[1][2][1]:
      if df>200 then
          res:=FastPrem_rc_rec (mp1, mp2, var, op(rp1)[1][1]):
      else
          res:=PlainPrem_rc_rec(mp1, mp2, var, op(rp1)[1][1]):
      end if:
      return res:
   end proc:



#------------------------------------------------------------------------------
# Function:
#    { Recdendegrpoly }
# Briefly:
#    { To compute the degree of the inputer Recden polynomial in the give variable}
# Calling sequence:
#    { Recdendegrpoly(rpoly1, v1)}
# Input:
#    { rpoly1: A recden polynomial.}
#    { v1: A variable.}
# Output:
#    { The degree of 'rpoly1' in 'v1'.}
#------------------------------------------------------------------------------
   Recdendegrpoly :=proc (rpoly1, v1)
      local mpoly;
      if (Algebraic:-RecursiveDensePolynomials:-isconstant(rpoly1)) then
         return 0:
      end if:
      mpoly := Algebraic:-RecursiveDensePolynomials:-rpoly(rpoly1):
      #Algebraic:-RecursiveDensePolynomials:-degrpoly(rpoly1, v1):
      return degree(mpoly, v1):
   end proc:
 end module;
