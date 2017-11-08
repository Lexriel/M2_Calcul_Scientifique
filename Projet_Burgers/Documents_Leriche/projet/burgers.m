% burgers - Programme to solve burgers equation. 
%
% There are two choices for the method :
%
%       n = 1   gives the Lax Wendroff scheme
%       n = 2   gives the two-step (Lax Wendroff) method
%
clear; help burgers;
global nu n
%
%   Input parameters
%
method = [@laxwend; @twostep];
n_method = input('Enter the method (1 or 2) -');
dx = input('Enter the grid spacing along the x-axis -');
cfl = input('Enter the Courant number -');
%
%   Set up grid and initial conditions
%
time = 0.;
tend = 0.2;
xend = 3*tend;
n = ceil(xend/dx);
dt = cfl*dx;
nu = dt/dx;
x = ((1:n) - 1)*dx;
uold = feval(@initial,x);
uplot1 = uold;
uexact1 = uold;
time1 = 0.;
time4 = 1.;
ntime1 = floor(tend/(3*dt));
ntime2 = 2*ntime1;
i = 0;
%
%   Iterate the equations up to t=tend
%
while time <= tend    
    unew = feval(method(n_method),uold);
    time = time + dt;
    unew(1) = 0; unew(n) = 0;
    uold = unew;    
    i = i + 1;
    if i == ntime1
        uplot2 = unew;
        time2 = time;
    end
    if i == ntime2
        uplot3 = unew;
        time3 = time;
    end    
end
uplot4 = unew;
subplot(221)
    plot(x,uplot1,'o')
    axis([0. xend 0. 1.1])
subplot(222)
    plot(x,uplot2,'o')
    axis([0. xend 0. 1.1])
subplot(223)
    plot(x,uplot3,'o')
    axis([0. xend 0. 1.1])
subplot(224)
    plot(x,uplot4,'o')
    axis([0. xend 0. 1.1])
subplot(111)
return;