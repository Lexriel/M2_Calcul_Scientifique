function [eb,hb] = q2beuler(tspan, y0, varargin)

L=tspan(2)-tspan(1);

for k=1:10
    Nh=10*k;
    h1=L/Nh;

    t(1)=tspan(1);
    u(1)=y0;

for i = 1:Nh
    r=u(i);
    r1=u(i)+10; %juste pour être sûr de faire au moins une boucle while
    
    while (abs(r-r1) > 1e-8)
        g=r-h1*cos(2*r)-u(i);
        gprime=1+2*h1*sin(2*r);
        r1=r;
        r = r - g / gprime ;
    end
    u(i+1)=r;
    
end

    
u(Nh+1)
eb(k)=abs(u(Nh+1)-0.650880168023008);
hb(k)=L/(10*k);

end

plot(log(hb),log(eb));
grid
xlabel('h')
ylabel('error')
p=(log(eb(10))-log(eb(1)))/(log(hb(10))-log(hb(1)));
disp('la valeur de p est:')
disp(p)

end