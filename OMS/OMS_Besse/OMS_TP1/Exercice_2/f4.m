function [Vf]=f4(X)
  x=X(1);
  y=X(2);
  z=X(3);
  
  % f est le vecteur suivant:
  f1=3*x-cos(y*z)-0.5;
  f2=x^2-81*(y+0.1)^2+sin(z)+1.06;
  f3=exp(-x*y)+20*z+(10*pi-3)/3;
  
  Vf=[f1;f2;f3];