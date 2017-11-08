function [u] = feuler6(f, tspan, y0, Nh, varargin)

% Définir f ainsi: f=@(u,t) [fonction de u(1), u(2), u(3) et t;idem;idem]

t=linspace(tspan(1),tspan(2),Nh+1);
h=t(2)-t(1);

u=zeros(6,1);
u=y0;

for i = 1:Nh
    u=u+h*f(u,t(i)); 
end