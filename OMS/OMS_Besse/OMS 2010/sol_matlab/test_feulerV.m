odefun=@(t,X) fun_ex6(X);
tspan=[0;25];
y0=[0;1;0;.8;0;1.2];
Nh=100000;
%[t,u]= feulerV(odefun ,tspan ,y0 ,Nh);
Nh=20000;
%[t,u]= beulerV(odefun ,tspan ,y0 ,Nh);
[t,u]= crankcV(odefun ,tspan ,y0 ,Nh);
plot3(u(1,:),u(2,:),u(3,:));
axis equal

[t1 ,y1]= ode23(@fvinc ,tspan ,y0');
[t2 ,y2]= ode45(@fvinc ,tspan ,y0');
figure(2); plot3(y1(:,1),y1(:,2),y1(:,3));
figure(3); plot3(y2(:,1),y2(:,2),y2(:,3));
options=odeset('RelTol',1.e-04);
[t,y]=ode45(@fvinc,tspan,y0,options);
figure(4); plot3(y(:,1),y(:,2),y(:,3));
