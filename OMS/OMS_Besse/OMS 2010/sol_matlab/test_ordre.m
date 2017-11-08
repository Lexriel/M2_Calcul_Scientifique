clear all;
tspan=[0 ,1];
y0=0;
f=@(t,y) cos(2*y);
df=@(t,y) -2*sin(2*y);
solex=@(t) 0.5*asin((exp(4*t)-1)./(exp(4*t)+1));
Nh=10;
%[t,ufe]= feuler(f,tspan,y0,Nh);
%[t,ufe]= beuler(f,tspan,y0,Nh);
%[t,ufe]= crankc(f,tspan,y0,Nh);
[t,ufe]= ode45(f,tspan,y0);
v1=abs(ufe(end)-solex(tspan(end)));
p=zeros(9,1);
for j=2:10
    Nh=Nh*2;
    %[t,ufe]= feuler(f,tspan,y0,Nh);
    %[t,ufe]= beuler(f,tspan,y0,Nh);
    %[t,ufe]= crankc(f,tspan,y0,Nh);
    [t,ufe]= ode45(f,tspan,y0);
    v2=abs(ufe(end)-solex(tspan(end)));
    p(j-1)=log(v1/v2)/log(2);
    v1=v2;
end
