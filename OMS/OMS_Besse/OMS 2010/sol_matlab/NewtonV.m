function [y,cpt]=NewtonV(f,df,y0,eps,itermax)
% Newton method for vector function
% f : function
% df : jacobian matrix
  
yk=y0;
cpt=1;
test=1;

while test
  res=df(yk)\f(yk);
  ykp=yk-res;
  yk=ykp;
  cpt=cpt+1;
  test=(norm(f(yk))>eps)&&(cpt<itermax);
end

y=ykp;