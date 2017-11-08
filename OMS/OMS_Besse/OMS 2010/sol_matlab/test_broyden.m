f=@(X) fun_f(X);
[yN,iter]=Broyden(f,[0.1;0.1;-0.1],1e-10,50);
fprintf('The result is [%f,%f,%f] after %d iterations\n',yN(1),yN(2),yN(3),iter);
