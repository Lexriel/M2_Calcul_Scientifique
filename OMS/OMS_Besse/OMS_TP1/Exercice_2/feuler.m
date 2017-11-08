function [t,u] = feuler(f, tspan, y0, Nh, varargin)


L=tspan(2)-tspan(1);
h=L/Nh;

t(1)=tspan(1);
u(1)=y0;


for i = 1:Nh
    t(i+1)=t(i)+h;
    u(i+1)=u(i)+h*f(u(i)); 
end

plot(t,u);

end