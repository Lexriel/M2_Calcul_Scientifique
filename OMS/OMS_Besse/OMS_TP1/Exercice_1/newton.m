function y=newton(f,df,y0,eps,itermax)

z=y0;
cpt=1;
test=1;

while test

    z=z-f(z)/df(z);
    cpt=cpt+1;
    test=(abs(f(z))>eps)&&(cpt<itermax);
end
y=z;