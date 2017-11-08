f=@(X) fun_f(X);
df=@(X) fun_Jacf(X);
[yN,iter]=NewtonV(f,df,[0.1;0.1;-0.1],1e-10,50);
fprintf('The result is [%f,%f,%f] after %d iterations\n',yN(1),yN(2),yN(3),iter);
