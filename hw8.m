clc
clear all

%% sys
A = [0 1;0 0];
C = [1 0];
%% parameters
x01 = [1; 1];
mean_1 = 1;
std_1 = 0.02;
x0 = [mean_1 + std_1*rand(1); mean_1 + std_1*rand(1)];

eps1 = 0; %case 1
eps2 = 0.1; %case 2
eps3 = 0.4; %case 3
eps = eps3;

N=10;
t0 = 0;
tf = 1;
dt = 0.01;
T = tf / dt;
t_all = linspace(t0, tf, T+1);
mean = pi / 2;
std = pi / 8;
phi = std.*randn(T+1, 1) + mean;

% Create empty data
x_total=zeros(N,2,T+1);
y_total=zeros(N,1,T+1);
x_bar_total =zeros(N,2,T+1);
x_star1_total =zeros(N,2,T+1);
x_star2_total =zeros(N,2,T+1);
error_bar_total=zeros(N,1,T+1);
error_star1_total=zeros(N,1,T+1);
error_star2_total=zeros(N,1,T+1);
%% Create x(t) y(t)

for i =1:N
    [x,y]=xtyt_fun(t_all,T,x0,eps,phi);
    x_total(i,:,:)=x;
    y_total(i,:,:)=y;
end
%% Generate estimation 
for i=1:N
    [x_bar,error_bar]=x_bar_fun(x0,T,y,dt,x);
    [x_star1,error_star1]=x_star_1_fun(x0,T,t_all,phi,eps,tf,x);
    [x_star2, error_star2]=x_star2_fun(x0,T,t_all,x);

    x_bar_total(i,:,:) = x_bar;
    x_star1_total(i,:,:) = x_star1;
    x_star2_total(i,:,:) = x_star2;
    error_bar_total(i,:,:) = error_bar;
    error_star1_total(i,:,:) = error_star1;
    error_star2_total(i,:,:) = error_star2;
end

%% Plotting for the last case


hold on
plot(t_all(1:T), error_star1(1,1:T))
plot(t_all(1:T), error_star2(1,1:T))
plot(t_all(1:T), error_bar(1,1:T))
xlim([0,1])
set(gca, 'YScale', 'log')
legend('$\bar{e}$','$e^{*1}$','$e^{*2}$','Interpreter','latex')
xlabel('time')
ylabel('value')
title('Last run error vs t,$\epsilon = 0.4$, mean = $\frac{\pi}{2}$, $\sigma = \frac{\pi}{8}$','Interpreter','latex')
s=2;
%%

subplot(2,1,1)
hold on
plot(t_all(1:T), x(1,1:T))
plot(t_all(1:T), x_star1(1,1:T))
plot(t_all(1:T), x_star2(1,1:T))
plot(t_all(1:T), x_bar(1,1:T))
xlim([0,1])
ylim([0,2.5])
legend('x','$\bar{x}$','$x^{*1}$','$x^{*2}$','Interpreter','latex')
xlabel('time')
ylabel('value')
title('x1 vs t,$\epsilon = 0.4$, mean = $\frac{\pi}{2}$, $\sigma = \frac{\pi}{8}$','Interpreter','latex')


subplot(2,1,2)
hold on
plot(t_all(1:T), x(2,1:T))
plot(t_all(1:T), x_star1(2,1:T))
plot(t_all(1:T), x_star2(2,1:T))
plot(t_all(1:T), x_bar(2,1:T))
xlim([0,1])
ylim([-30,30])
legend('x','$\bar{x}$','$x^{*1}$','$x^{*2}$','Interpreter','latex')
xlabel('time')
ylabel('value')
title('x2 vs t,$\epsilon = 0.4$, mean = $\frac{\pi}{2}$, $\sigma = \frac{\pi}{8}$','Interpreter','latex')
s=2;
%% Plotting error average
error_bar_avg = zeros(1,T+1);
error_star1_avg = zeros(1,T+1);
error_star2_avg = zeros(1,T+1);

for i =1:N
    error_bar_avg = error_bar_avg + reshape(error_bar_total(i,:,:),[1,T+1]);
    error_star1_avg = error_star1_avg + reshape(error_star1_total(i,:,:),[1,T+1]);
    error_star2_avg = error_star2_avg + reshape(error_star2_total(i,:,:),[1,T+1]);
end
error_bar_avg = error_bar_avg/N;
error_star1_avg = error_star1_avg/N;
error_star2_avg = error_star2_avg/N;



hold on 
plot(t_all,error_bar_avg);
plot(t_all,error_star1_avg);
plot(t_all,error_star2_avg);
xlabel('time')
ylabel('value')
xlim([0,1]);
set(gca, 'YScale', 'log')
legend('$\bar{e}$','$e^{*1}$','$e^{*2}$','Interpreter','latex')
title('Averaged error (10 runs) vs t,$\epsilon = 0.4$, mean = $\frac{\pi}{2}$, $\sigma = \frac{\pi}{8}$','Interpreter','latex')
s=2;
%% Utility functions




% create x(t) y(t)
function [x,y]=xtyt_fun(t_all,T,x0,eps,phi)

y = zeros(1,(T+1));
x = zeros(2 , T+1);
for i = 1:T+1
    t = t_all(i);
    xt = [1  t; 0 1] * x0;
    x(:, i) = xt;
    y(1,i) = [1 0] * xt + eps * sin(50 * t + phi(i));
end

end

% x_bar
function [x_bar,error_bar]=x_bar_fun(x0,T,y,dt,x)

error_bar = zeros(1, T+1);
x_bar = zeros(2, T+1);
for i = 1:T+1
    x_bar(1, i) = y(i); 
    if i < T+1
        x_bar(2, i) = (y(i+1) - y(i)) / dt;
    end
    error_bar(i) = norm(x_bar(:, i) - x(:, i), 2);
end

end

%x_star1
function [x_star1,error_star1]=x_star_1_fun(x0,T,t_all,phi,eps,tf,x)
error_star1 = zeros(1, T+1);
x_star1 = zeros(2, T+1);
for i =1:T+1
    t = t_all(i);
    phi_t = phi(i);
    W = [1 - t, 0.5*t^2 - t + 0.5; ...
     0.5*t^2 - t + 0.5, -t^3/3 + t^2 - t + 1/3];
    
    % y(tau) = x0(1) .* (1 + tau) + eps .* sin(50.*tau + phi_t)
    fun1 = @(tau) x0(1) .* (1 + tau) + eps .* sin(50.*tau + phi_t); 
    fun2 = @(tau) (tau - t) .* x0(1) .* (1 + tau) + eps .* sin(50*tau + phi_t); 
    % fun1 = @(tau) x0(1) .* (1 + tau); 
    % fun2 = @(tau) (tau - t) .* x0(1) .* (1 + tau);
    int1 = integral(fun1, t , tf);
    int2 = integral(fun2, t , tf);

    x_star1(:, i) = W^(-1) * [int1 ; int2];
    error_star1(i) = norm(x_star1(:, i) - x(:, i), 2);
end

end



% x_star2
function [x_star2, error_star2]=x_star2_fun(x0,T,t_all,x) 

error_star2 = zeros(1, T+1);
x_star2 = zeros(2, T+1);
for i =1:T+1
    t = t_all(i);
    W = [1 0.5; 0.5 1/3];
    x_star2(:, i) = [1 t;0 1] * W^(-1) * [3/2 * x0(1) ; 5/6 * x0(1)];
    error_star2(i) = norm(x_star2(:, i) - x(:, i), 2);
end

end