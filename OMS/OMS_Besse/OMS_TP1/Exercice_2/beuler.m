function [t,u] = beuler(tspan, y0, Nh, varargin)

t=linspace(tspan(1),tspan(2),Nh+1);
h=t(2)-t(1);
u(1)=y0;

for i = 1:Nh
    r=u(i);
    r1=u(i)+10;
    
    while (abs(r-r1) > 1e-8)
        g=r-h*cos(2*r)-u(i);
        gprime=1+2*h*sin(2*r);
        r1=r;
        r = r - g / gprime ;
    end
    u(i+1)=r;
    
end

plot(t,u);

end
    