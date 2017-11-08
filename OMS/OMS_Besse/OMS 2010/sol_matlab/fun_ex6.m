function [V]=fun_ex6(X);
  x=X(1);   y=X(2);  z=X(3);
  vx=X(4); vy=X(5); vz=X(6);
  V=zeros(6,1);
  V(1)=vx; V(2)=vy; V(3)=vz;
  g=9.8;
  m=1;
  F=[0;0;-g*m];
  H=diag([2;2;2]);
  W=[vx;vy;vz];
  dphi=[2*x;2*y;2*z];
  lambda=(m*W'*(H*W)+dphi'*F)/norm(dphi)^2;
  
  V(4)=(F(1)-lambda*dphi(1))/m;
  V(5)=(F(2)-lambda*dphi(2))/m;
  V(6)=(F(3)-lambda*dphi(3))/m;
  