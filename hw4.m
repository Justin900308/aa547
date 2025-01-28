clc 
clear all

A1=[2 -1;-1 2];
[V,D]=eig(A1)
[P,J]=jordan(A1)

norm1=zeros(1,101);
norm2=zeros(1,101);
t=linspace(0,2,101);
for i = 1:length(t)
    norm1(i)=exp(3*t(i));
    norm2(i)=norm(expm(t(i)*A1));
    

end
hold on
plot(t,norm1,'r.',MarkerSize=20)
plot(t,norm2,'g.-')
xlabel('t(sec)')
ylabel('value')
leg1 = legend('$\|V\|\|V^{-1}\| \exp^{t\zeta(A)}d \max_k \frac{t^k}{k!}$','$\|\exp^{tA}\|$');
set(leg1,'Interpreter','latex');


norm(V^(-1))
%%
clc 
clear all
A2=[-1 2;0 -1];
[V2,D2]=eig(A2)
[P2,J2]=jordan(A2)



norm1=zeros(1,101);
norm2=zeros(1,101);
t=linspace(0,5,101);
for i = 1:length(t)
    norm1(i)=norm(P2)*norm(P2^(-1))*exp(-t(i))*2*max(1,t(i)/1);
    norm2(i)=norm(expm(t(i)*A2));
    

end
hold on
plot(t,norm1,'r.',MarkerSize=20)
plot(t,norm2,'g.-')
xlabel('t(sec)')
ylabel('value')
leg1 = legend('$\|V\|\|V^{-1}\| \exp^{t\zeta(A)}d \max_k \frac{t^k}{k!}$','$\|\exp^{tA}\|$');
set(leg1,'Interpreter','latex');

%%
clc 
clear all
A3=[-0.5 1 0;0 -0.5 1;0 0 -0.5];
[V3,D3]=eig(A3)
[P3,J3]=jordan(A3)



norm1=zeros(1,101);
norm2=zeros(1,101);
t=linspace(0,10,101);
for i = 1:length(t)
    array=[1,t(i)/1,t(i)^2/2];
    norm1(i)=norm(P3)*norm(P3^(-1))*exp(-0.5*t(i))*3*max(array);
    norm2(i)=norm(expm(t(i)*A3));
    

end
hold on
plot(t,norm1,'r.',MarkerSize=20)
plot(t,norm2,'g.-')
xlabel('t(sec)')
ylabel('value')
leg1 = legend('$\|V\|\|V^{-1}\| \exp^{t\zeta(A)}d \max_k \frac{t^k}{k!}$','$\|\exp^{tA}\|$');
set(leg1,'Interpreter','latex');