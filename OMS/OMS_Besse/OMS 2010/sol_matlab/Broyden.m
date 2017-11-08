function [y,cpt]=Broyden(f,y0,eps,itermax)
% Broyden method for vector function
% f : function

  N=length(y0);
  h=1.e-5;
  Ak=zeros(N,N);
  for j=1:N
    y0p=y0; y0p(j)=y0p(j)+h;
    y0m=y0; y0m(j)=y0m(j)-h;
    
    Ak(:,j)=f(y0p)-f(y0m);
  end
  Ak=Ak/2/h;

  yk=y0;
  cpt=1;
  test=1;
  
  while test
    res=Ak\f(yk);
    ykp=yk-res;
    u=f(ykp)-f(yk);
    s=ykp-yk;
    Ak=Ak+(u-Ak*s)*(s')/norm(s,2)^2;
    yk=ykp;
    cpt=cpt+1;
    test=(norm(f(yk))>eps)&&(cpt<itermax);
  end
  
  y=ykp;