#################################################
# REGULARCHAINS library                                #
# File: FastTriade.mpl                          #
# Last update: October 2007                     #
# Contact:                                      #
#    Xin Li <xli96@csd.uwo.ca>                  #
#               and                             #
#    Marc Moreno Maza <moreno@csd.uwo.ca>       #
#################################################

#------------------------------------------------------------------------------
# Function:
# Briefly:
# Calling sequence:
# {
# Input:
# {}
# Assumptions:
# {}
# Output:
#------------------------------------------------------------------------------
TRDFastTriangularize := proc(F, p, vars)
    local R, RR, f1, f2, MSoutput, item, newres, ec, rc, lpol, pol, f1mvar;

    ###############
    # Checking in #
    ###############
    if nops(F) <> 2 then 
       error("Invalid input: arg #1 must a list of two polynomials"); 
    end if;
    if (nops(vars) <> 2) or (p = 0) then 
       error("Invalid input: arg #2 must a bivariate polynomial ring over a finite field"); 
    end if;

    ###########################################################
    #### Pure C implementation, Feb 2011, WP
    ###########################################################

    R := TRDpolynomial_ring(vars, {}, p);
    RR := modpn:-PolynomialRing(p, vars);
    MSoutput := modpn:-BivariateSolve(F[1], F[2], RR); 

    newres := [];
    ec := TRDempty_regular_chain();
    for item in MSoutput do
        rc := TRDlist_pretend_regular_chain2(item, ec, R);
        newres := [rc, op(newres)];
    end do;
    return newres;

    ##########################################################
    #### THE FOLLOWING CODE IS NOT CALLED ANY MORE
    ##########################################################

    ##############
    # Setting up #
    ##############
    RR := modpn:-PolynomialRing(p, vars);
    f1 := modpn:-PolynomialConvertIn(RR, F[1]);
    f2 := modpn:-PolynomialConvertIn(RR, F[2]);
    f1mvar:=modpn:-RecdenConnector:-RecdenMainVariable(f1:-RecdenPoly);
    if (modpn:-RecdenConnector:-Recdendegrpoly(f1:-RecdenPoly,f1mvar) <  
        modpn:-RecdenConnector:-Recdendegrpoly(f2:-RecdenPoly,f1mvar)) then         
       (f1, f2) := (f2, f1)
    end if:

    ##############################
    # Calling the modular solver #
    ##############################
    MSoutput:= TRDModularSolve2(RR, f1:-RecdenPoly, f2:-RecdenPoly);

    ###########################################
    # Converting out to RegularChains representation #
    ###########################################
    R := TRDpolynomial_ring(vars,{},p);

    ec := TRDempty_regular_chain();

    newres := [];

    for item in MSoutput do
        lpol := [seq(TRDrecden2Maple(pol, R), pol in item)];
        rc := TRDlist_pretend_regular_chain2(lpol, ec, R);
        newres := [rc, op(newres)];
    end do;

    return(newres);

end proc:
