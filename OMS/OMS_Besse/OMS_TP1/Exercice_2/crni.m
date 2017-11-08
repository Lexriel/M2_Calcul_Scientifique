function [t,u] = crni(tspan, y0, Nh, varargin)

L=tspan(2)-tspan(1);
h=L/Nh;

t(1)=tspan(1);
u(1)=y0;

for i = 1:Nh
    t(i+1)=t(i)+h;
    r=u(i);
    r1=u(i)+10;
    
    while (abs(r-r1) > 1e-8)
        g=r-h/2*cos(2*r)-u(i)-h/2*cos(2*u(i));
        gprime=1+h*sin(2*r);
        r1=r;
        r = r - g / gprime ;
    end
    u(i+1)=r;
    
end

plot(t,u);

end
    