function [p1,p2]=q3ode(odefun)



e1=zeros(5,1);
e2=zeros(5,1);
h=zeros(5,1);

for i=-5:-1
    h(i+6)=2^i;
    
    options = odeset('MaxStep', h(i+6), 'InitialStep', h(i+6), 'AbsTol',1.0);
    [t1,y1]=ode23(odefun, (0:1000)/1000, 0, options);
    
    [t2,y2]=ode45(odefun,(0:1000)/1000, 0, options);

    a1=y1(end);
    a2=y2(end);

    e1(i+6)=abs(a1-0.650880168023008);
    e2(i+6)=abs(a2-0.650880168023008);
    
end   

plot(log(h),log(e1),log(h),log(e2))
grid on
xlabel('ln(h)')
ylabel('ln(e1) et ln(e2)')

disp('la valeur de p1 est:')
p1=( log(e1(5)) - log(e1(1)) ) / ( log(h(5)) -log(h(1)) );
disp(p1)

disp('la valeur de p2 est:')
p2=( log(e2(5)) - log(e2(1)) ) / ( log(h(5)) -log(h(1)) );
disp(p2)