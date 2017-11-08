function [y]=Secant(f,y0,y1,eps,itermax)
yk=zeros(itermax+1,1);
yk(1)=y0;
yk(2)=y1;
cpt=2;
test=1;

while test
    yk(cpt+1)=yk(cpt)-f(yk(cpt))*(yk(cpt)-yk(cpt-1))/...
        (f(yk(cpt))-f(yk(cpt-1)));
    cpt=cpt+1;
    test=(abs(f(yk(cpt)))>eps)&&(cpt<itermax);
end
y=zeros(cpt,1);
y=yk(1:cpt);