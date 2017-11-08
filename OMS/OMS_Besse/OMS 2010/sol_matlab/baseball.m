h=0.01;
x=0;y=0;z=1;
v0=38;
B=4.1e-4;
g=9.81;
phi=1;
om=180*1.047198;
vx=v0*cos(phi);
vy=0;
vz=v0*sin(phi);
test=1;
X=zeros(10000,1);Y=X;Z=X;
X(1)=x;Y(1)=y;Z(1)=z;
VX=zeros(10000,1);VY=VX;VZ=VX;
VX(1)=vx;VY(1)=vy;VZ(1)=vz;

cpt=1;

while test
    v=sqrt(vx^2+vy^2+vz^2);
    F=0.0039+0.0058/(1+exp((v-35)/5));
    vxp=vx+h*(  -F*v*vx+B*om*(vz*sin(phi)-vy*cos(phi)));
    vyp=vy+h*(  -F*v*vy+B*om*vx*cos(phi));
    vzp=vz+h*(-g-F*v*vz-B*om*vx*sin(phi));
    vx=vxp;vy=vyp;vz=vzp;
    x=x+h*vx;
    y=y+h*vy;
    z=z+h*vz;
    test=z>1;
    cpt=cpt+1;
    X(cpt)=x;Y(cpt)=y;Z(cpt)=z;
    VX(cpt)=vx;VY(cpt)=vy;VZ(cpt)=vz;
end