function [y]=Newton(f,df,y0,eps,itermax)
yk=zeros(itermax+1,1);
yk(1)=y0;
cpt=1;
test=1;

while test

    yk(cpt+1)=yk(cpt)-f(yk(cpt))/df(yk(cpt));
    cpt=cpt+1;
    test=(abs(f(yk(cpt)))>eps)&&(cpt<itermax);
end
y=zeros(cpt,1);
y=yk(1:cpt);