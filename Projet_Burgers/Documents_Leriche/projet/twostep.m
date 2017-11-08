function unew = twostep(uold)
global nu n
flux(1:n) = uold(1:n).*uold(1:n)/2.;
dflux(1:n-1) = nu*(flux(2:n) - flux(1:n-1))/2.;
uhalf(1:n-1) = 0.5*(uold(1:n-1) + uold(2:n))...
    -0.5*dflux(1:n-1);
flux(1:n-1) = uhalf(1:n-1).*uhalf(1:n-1)/2.;
dflux(1:n-2) = nu*(flux(2:n-1) - flux(1:n-2));
unew(2:n-1) = uold(2:n-1) - dflux(1:n-2);
return;