function y=newton3(f,df,eps,itermax)

% X and f are vectors, df is the jacobian matrix of f
s=[0.1;0.1;-0.1];
cpt=1;
test=1;

while test
    x=s(1);
    y=s(2);
    z=s(3);
    
    s=s-df(x,y,z)\f(x,y,z);
    cpt=cpt+1;
    test=(norm(f(s(1),s(2),s(3)))>eps)&&(cpt<itermax);
end
y=s;