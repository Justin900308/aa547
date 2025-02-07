clc
clear all


A=[-1 2;0 -1];
Q=eye(2);
P = lyap(A,Q);
[V,diag] = eig(P);
cond_P = norm(P)*norm(P^(-1));
C=sqrt(cond_P);
lambda_min=0.5*min(eig(P^(-0.5)*Q*P^(-0.5)));

%%


A2=[-1 2;0 -1];
[V2,D2]=eig(A2)
[P2,J2]=jordan(A2)



norm1=zeros(1,101);
norm2=zeros(1,101);
norm3=zeros(1,101);
t=linspace(0,10,201);
for i = 1:length(t)
    norm1(i)=norm(P2)*norm(P2^(-1))*exp(-t(i))*2*max(1,t(i)/1);
    norm2(i)=norm(expm(t(i)*A2));
    norm3(i)=C*exp(-lambda_min*t(i));

end
hold on
plot(t,norm1,'r.',MarkerSize=20)
plot(t,norm2,'g.-')
plot(t,norm3,'bo')
xlabel('t(sec)')
ylabel('value')
leg1 = legend('$\|V\|\|V^{-1}\| \exp^{t\zeta(A)}d \max_k \frac{t^k}{k!}$','$\|\exp^{tA}\|$', 'Lyapunove');
set(leg1,'Interpreter','latex');