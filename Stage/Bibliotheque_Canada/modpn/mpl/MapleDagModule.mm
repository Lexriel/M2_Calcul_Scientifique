MapleDagModule := module()

    # exports.
export MapleDag2CData, Verify, ParsePoly, FinalList2Array, isListArrayEqual;

    # locals.
local ListOfSum, ListOfProd, ParseExpr, Seq2Vec, ParseP,   MyList2Array;

    # options.
option package;

    #  Procedure: ListofSum.
    #  Parsing a "sum" expression
    #  INPUT:     res, the pervious result "list of lists" to
    #  which  this procedure concatinates.
    #  L, the list of operands of the sum expression.
    #  loc: index
    #  res: a list meant to encode a SLP
    #  OUTPUT:  r, the concatinated new list of lists.
    ListOfSum := proc(res, L , loc)
        local i, l, l2, n, r;
        r:=res;
        l:=loc;
        n:=nops(L);
        for i from 1 to n do
            r:=ParseExpr(r, L[i]);
            l2:=nops(r);
            if l>0  then
                # r:=[cat(convert(l, string), " + ",
                # convert(l2, string)) ,op(r)];
                r:=[[4, l, l2], op(r)];
                l2:=l2+1;
            end if;
            l:=l2;
        end do;
        return r;
    end proc;

    # similar to ListOfSum except that the operator is
    # encode by "5" instead of "4".
    ListOfProd := proc(res, L , loc)
        local i, l, l2, n, r;
        r:=res;
        l:=loc;
        n:=nops(L);
        for i from 1 to n do
            r:=ParseExpr(r, L[i]);
            l2:=nops(r);
            if l>0  then
                #r:=[cat(convert(l, string), " * ",
                #convert(l2, string)) ,op(r)];
                r:=[[5,l,l2], op(r)];
                l2:=l2+1;
            end if;
            l:=l2;
        end do;
        return r;
    end proc;

    # Parse the expression.
    # the expression maybe any five of following types:
    # integer, symbol, +, *, ^
    #   num                                                0
    #   var                                               1
    #   power      var^num =                              2
    #   power   addr^num=                                 3
    #    \"+\" operands must be addr                      4
    #    \"*\" operands must be addr                      5
    #     addr                                            6


    ParseExpr := proc (result, expr)
        local r;
        r := result;
        if type(expr, integer) then
            r := [[0, expr], op(r)]
        end if;
        if type(expr, symbol) then
            r := [[1, expr], op(r)]
        end if;
        if type(expr, `+`) then
            r := ListOfSum(r, [op(expr)],  0)
        end if;
        if type(expr, `*`) then
            r := ListOfProd(r, [op(expr)], 0)
        end if;
        if type(expr, `^`) then
            r := [[23, op(1, expr), op(2, expr)], op(r)]
        end if;
        return r;
    end proc;

    # convert Sequence to vector.
    Seq2Vec := proc(V, list)
        local i, n;
        n:=nops(list);
        for i from 1 to n do  V[i]:=list[i]; end do;
        return V;
    end proc;

    # inner funciton of ParsePoly.
    ParseP := proc (P)
        local V, Vlocation, Vsymbol, exprSeq, i, j, l, m, n, op2, rec, result;
        result := [];

        #print("whattype(P)", whattype(P));
        exprSeq := codegen:-optimize(P, 'tryhard');
        #print("exprSeq", exprSeq);

        n := nops([exprSeq]);
        V := Array(1 .. n);
        V := Seq2Vec(V, [exprSeq], 1);
        Vsymbol := Array(1 .. n);
        Vlocation := Array(1 .. n);
        l := 0;
        for i to n do
            Vsymbol[i] := op(1, V[i]);
            op2 := op(2, V[i]);
            result := ParseExpr(result, op2);
            l := nops(result);
            Vlocation[i] := l
        end do;
        m := nops(result);
        for i to m do
            for j to n do
                if result[i][1] = 1 and result[i][2] = Vsymbol[j] then
                    #result[i][1] := 6; result[i][2] := Vlocation[j];
                    rec:= result[i]: rec[1]:=6: rec[2]:=Vlocation[j]:
                    result:=subsop(i=rec,result);
                    break
                end if;
                if result[i][1] = 23 and result[i][2] = Vsymbol[j] then
                    #result[i][1] := 3; result[i][2] := Vlocation[j];
                    rec:= result[i]: rec[1]:=3: rec[2]:=Vlocation[j]:
                    result:=subsop(i=rec,result);
                end if
            end do;
            if result[i][1] = 23 then
                #result[i][1] := 2
                rec:=result[i]: rec[1]:=2:
                result:=subsop(i=rec, result);
            end if
        end do;
        return result;
    end proc;


    # Parse a Maple Polynomial (dag) to list of lists type.
    # P is a polynomial
    # Returns an SLP encoding of the input polynomial P
    ParsePoly := proc(P)
        local r;
        r:=ParseP(P);
        return ListTools:-Reverse(r);
    end proc;

    # inner function of FinalList2Array
    MyList2Array := proc (varL, A, R)
        local inx, item, n, r, varLszAddone, varno;
        n := nops(R);
        varLszAddone := nops(varL)+1;
        for inx to n do
            r := R[inx];
            if r[1] = 0 then
                A[inx, 1] := 0; A[inx, 2] := r[2]
            end if;
            if r[1] = 1 or r[1] = 2 then
                A[inx, 1] := r[1];
                varno := 0;
                for item in varL do
                    varno := varno+1;
                    if r[2] = item then
                        A[inx, 2] := varLszAddone-varno;
                        break
                    end if
                end do;
                if r[1] = 2 then A[inx, 3] := r[3] end if
            end if;
            if r[1] = 6 then A[inx, 1] := 6; A[inx, 2] := r[2] end if;
            if r[1] = 3 or r[1] = 4 or r[1] = 5 then
                A[inx, 1] := r[1]; A[inx, 2] := r[2]; A[inx, 3] := r[3]
            end if
        end do;
        return A;
    end proc;

    # Converting the list of list (TAC statements)
    # into Array type which will be directly
    # passed to C .
    FinalList2Array := proc (varL, L)
        local  n, A;
        n := nops(L);
        A := Array(1 .. n, 1 .. 3, 'order' = 'C_order', 'datatype' = 'integer[4]');
        A := MyList2Array(varL, A, L);
        return A;
    end proc;





    # converting a Maple polynomial to C data for constructing
    # C-level data.
#------------------------------------------------------------------------------
# Function:
#    { MapleDag2CData }
# Briefly:
#    { To convert the input Maple DAG polynomial into an immediate
#      encoding to be passed to C level, actually to an SLP}
# Calling sequence:
#    { MapleDag2CData(P, VarList)}
# Input:
#    { P: a Maple polynomial.}
#    { VarList: a variable list.}
# Output:
#    { A Vector encodes the input Maple DAG which will be decoded at C level to a C-DAG polynomial.}
#------------------------------------------------------------------------------
    MapleDag2CData:= proc(P, VarList)
        local L, V;
        L:=ParsePoly(P);
        V:=FinalList2Array(VarList, L);
        return V;
    end proc;








    # tester to test if Section#2's result is correct based on Sectoin#1's   result.
    isListArrayEqual := proc (L, A, Lvar)
        local Lij, dims, i, it, j, m, n, res, varLszAddone, varno;
        res := "Euqal"; dims := ArrayDims(A); n := op(dims[1])[2];
        varLszAddone := nops(Lvar)+1;
        for i to n do
            m := nops(L[i]);
            for j to m do
                Lij := L[i, j];
                if type(Lij, symbol) then
                    varno := 0;
                    for it in Lvar do
                        varno := varno+1;
                        if Lij = it then
                            Lij := varLszAddone-varno;
                            break
                        end if
                    end do
                end if;
                if A[i, j] <> Lij then
                    #lprint(i, j);
                    return "NOT EQUAL"
                end if
            end do
        end do;
        return res;
    end proc;






    # To verify if the two steps inside the conversion is consistent.
    # The first step, ParsePoly constructs a List of TAC statements.
    # The sceond step, FinalList2Array converting this List into Vector.

#------------------------------------------------------------------------------
# Function:
#    { Verify }
# Briefly:
#    { To verify the correctness of MapleDag2CData().}
# Calling sequence:
#    { }
# Input:
#    { }
# Output:
#    { }
#------------------------------------------------------------------------------
    Verify := proc(P, VarList)
        local L, V;
        L:=ParsePoly(P);
        V:=FinalList2Array(VarList, L);
        return isListArrayEqual(L, V, VarList);
    end proc;

end module;
