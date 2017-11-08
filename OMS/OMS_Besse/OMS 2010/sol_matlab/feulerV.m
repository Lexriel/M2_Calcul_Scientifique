function [t,u]= feulerV(odefun ,tspan ,y0 ,Nh ,varargin )
% FEULERV Solves differential system of equations using 
% the forward Euler method.
% [T,Y]= FEULERV(ODEFUN ,TSPAN ,Y0 ,NH) with TSPAN=[T0 ,TF]
% integrates the system of differential equations % y?=f(t,y) from time T0
% to TF with initial condition 
% Y0 using the forward Euler method on an equispaced 
% grid of NH intervals . 
% Function ODEFUN(T,Y) must return a vector , whose 
% elements hold the evaluation of f(t,y), of the 
% same dimension of Y. 
% Each row in the solution array Y corresponds to a 
% time returned in the column vector T. 
% [T,Y] = FEULERV(ODEFUN ,TSPAN ,Y0 ,NH ,P1 ,P2 ,...) passes 
% the additional parameters P1 ,P2 ,... to the function 
% ODEFUN as ODEFUN(T,Y,P1 ,P2 ...).

t=linspace(tspan(1),tspan(2),Nh)';
dt=t(2)-t(1);
u=zeros(length(y0),Nh);
u(:,1)=y0;
for j=2:Nh
    u(:,j)=u(:,j-1)+dt*odefun(t(j-1),u(:,j-1));
end
