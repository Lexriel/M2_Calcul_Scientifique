function [y] = beuler3(f, tspan, y0, Nh, eps, itermax)

t=linspace(tspan(1),tspan(2),Nh+1);
h=t(2)-t(1);

y=y0;

for i=1:Nh
    F=@(Y) Y-y-h*f(Y,t(i+1));
    y=broyden(F,y,h,eps,itermax);
end