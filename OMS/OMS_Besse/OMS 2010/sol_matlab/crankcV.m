function [t,u]= crankcV(odefun ,tspan ,y0 ,Nh ,varargin )
% CRANKC Solves differential equations using the Crank
% Nicolson method.
% [T,Y]= CRANKC(ODEFUN ,TSPAN ,Y0 ,NH) with TSPAN=[T0 ,TF]
% integrates the system of differential equations % y?=f(t,y) from time T0
% to TF with initial condition 
% Y0 using the Crank Nicolson method on an equispaced 
% grid of NH intervals . 
% Function ODEFUN(T,Y) must return a vector , whose 
% elements hold the evaluation of f(t,y), of the 
% same dimension of Y. 
% Each row in the solution array Y corresponds to a 
% time returned in the column vector T. 
% [T,Y] = CRANKC(ODEFUN ,TSPAN ,Y0 ,NH ,P1 ,P2 ,...) passes 
% the additional parameters P1 ,P2 ,... to the function 
% ODEFUN as ODEFUN(T,Y,P1 ,P2 ...).

  itermax=50;
  t=linspace(tspan(1),tspan(2),Nh)';
  dt=t(2)-t(1);
  u=zeros(Nh,1);
  u=zeros(length(y0),Nh);
  u(:,1)=y0;
  for j=2:Nh
    f=@(x) x-u(:,j-1)-0.5*dt*(odefun(t(j),x)+odefun(t(j-1),u(:,j-1)));
    [y,iter]=Broyden(f,u(:,j-1),1e-10,itermax);
    if (iter>itermax)
      fprintf('Maximum number of iteration reached in Broyden method\n');
      break;
    end
    u(:,j)=y;
    if (rem(j,1000)==0)
      fprintf('Iteration %d\n',j);
    end
  end