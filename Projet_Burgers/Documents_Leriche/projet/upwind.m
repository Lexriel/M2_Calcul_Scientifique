function unew = upwind(uold)
global nu n
flux(1:n) = uold(1:n).*uold(1:n)/2;
wave_speed = feval(speed(n_speed),x,time);
unew(2:n-1) = uold(2:n-1) - nu*wave_speed(2:n-1).*...
    (uold(2:n-1) - uold(1:n-2));
return;