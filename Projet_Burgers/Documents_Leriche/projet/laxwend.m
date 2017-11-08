function unew = laxwend(uold)
global nu n
flux(1:n) = uold(1:n).*uold(1:n)/2.;
speed(1:n-1) = nu*(uold(1:n-1) + uold(2:n))/2.;
d_flux(1:n-1) = nu*(flux(2:n) - flux(1:n-1))/2.;
unew(2:n-1) = uold(2:n-1) - (1. - speed(2:n-1)).*d_flux(2:n-1)...
    - (1. + speed(1:n-2)).*d_flux(1:n-2);
return;