function [ef,hf] = q2feuler(tspan, y0, varargin)

f=@(x) cos(2*x);
L=tspan(2)-tspan(1);

for k=1:10
    Nh=10*k;
    h1=L/Nh;

    t(1)=tspan(1);
    u(1)=y0;

    t(1)=tspan(1);
    u(1)=y0;


    for i = 1:Nh
        t(i+1)=t(i)+h1;
        u(i+1)=u(i)+h1*f(u(i)); 
    end

ef(k)=abs(u(Nh+1)-0.650880168023008);
hf(k)=L/(10*k);

end

plot(log(hf),log(ef));
grid
xlabel('h')
ylabel('error')
p=(log(ef(10))-log(ef(1)))/(log(hf(10))-log(hf(1)));
disp('la valeur de p est:')
disp(p)

end
    
    
