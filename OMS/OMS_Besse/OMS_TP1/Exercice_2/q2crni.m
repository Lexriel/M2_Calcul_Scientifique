function [e,h] = q2crni(tspan, y0, varargin)

L=tspan(2)-tspan(1);

for k=1:10
    Nh=10*k;
    h1=L/Nh;

    t(1)=tspan(1);
    u(1)=y0;

    for i = 1:Nh
        r=u(i);
    r1=u(i)+10;
    
    while (abs(r-r1) > 1e-8)
            g=r-h1/2*cos(2*r)-u(i)-h1/2*cos(2*u(i));
            gprime=1+h1*sin(2*r);
            r1=r;
            r = r - g / gprime ;
        end
        u(i+1)=r;
    
    end

% we compute the difference between u(Nh+1) and the real value of the
% solution at time Tf=1 here
u(Nh+1)
e(k)=abs(u(Nh+1)-0.650880168023008);
h(k)=L/(10*k);

end

plot(log(h),log(e));
grid
xlabel('h')
ylabel('error')
p=(log(e(10))-log(e(1)))/(log(h(10))-log(h(1)));
disp('la valeur de p est:')
disp(p)

end