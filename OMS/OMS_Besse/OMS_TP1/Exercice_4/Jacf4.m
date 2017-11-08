function [Jf]=Jacf4(X)
  x=X(1);
  y=X(2);
  z=X(3);
  
  % Jf est la matrice suivante:
  Jf1=[3,z*sin(y*z),y*sin(y*z)];
  Jf2=[2*x,-2*81*(y+0.1),cos(z)];
  Jf3=[-y*exp(-x*y),-x*exp(-x*y),20];
  
  Jf=[Jk1;Jk2;Jk3];