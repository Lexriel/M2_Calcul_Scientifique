f=@(x) tanh(x).*cos(x.^2)+x+2;
df=@(x) (1-tanh(x).^2)*cos(x.^2)-2*tanh(x).*sin(x.^2).*x+1;
xx=linspace(-3,-1,100)';
yN=Newton(f,df,-2.,1e-10,50);
yS=Secant(f,-2.,-1.5,1e-10,50);