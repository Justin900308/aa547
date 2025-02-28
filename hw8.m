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

eps1 = 0;
eps2 = 0.1;
eps3 = 0.4;
eps = eps2;

t0 = 0;
tf = 1;
dt = 0.01;
T = tf / dt;
t_all = linspace(t0, tf, T+1);
mean = pi / 2;
std = pi / 8;
phi = std.*randn(T+1, 1) + mean;
%% Create y(t)
y = zeros(1,(T+1));
x = zeros(2 , T+1);
for i = 1:T+1
    t = t_all(i);
    xt = [1  t; 0 1] * x0;
    x(:, i) = xt;
    y(1,i) = [1 0] * xt + eps * sin(50 * t + phi(i));
end
%% x_bar
error_bar = zeros(1, T+1);
x_bar = zeros(2, T+1);
for i = 1:T+1
    x_bar(1, i) = y(i); 
    if i < T+1
        x_bar(2, i) = (y(i+1) - y(i)) / dt;
    end
    error_bar(i) = norm(x_bar(:, i) - x(:, i), 2);
end
%% x_star1
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

%% x_star2
error_star2 = zeros(1, T+1);
x_star2 = zeros(2, T+1);
for i =1:T+1
    t = t_all(i);
    W = [1 0.5; 0.5 1/3];
    x_star2(:, i) = [1 t;0 1] * W^(-1) * [3/2 * x0(1) ; 5/6 * x0(1)];
    error_star2(i) = norm(x_star2(:, i) - x(:, i), 2);
end


%% Plotting





hold on
plot(t_all(1:T), error_star1(1,1:T))
plot(t_all(1:T), error_star2(1,1:T))
plot(t_all(1:T), error_bar(1,1:T))
xlim([0,1])
ylim([0,60])
legend('x star1','x star2','x bar')
xlabel('time')
ylabel('value')
title('x1 vs t,$\epsilon = 0$, mean = $\frac{\pi}{2}$, $\sigma = \frac{\pi}{8}$','Interpreter','latex')
s=2
%%

subplot(2,1,1)
hold on
plot(t_all(1:T), x(1,1:T))
plot(t_all(1:T), x_star1(1,1:T))
plot(t_all(1:T), x_star2(1,1:T))
plot(t_all(1:T), x_bar(1,1:T))
xlim([0,1])
ylim([0,2.5])
legend('x','x star1','x star2','x bar')
xlabel('time')
ylabel('value')
title('x1 vs t,$\epsilon = 0$, mean = $\frac{\pi}{2}$, $\sigma = \frac{\pi}{8}$','Interpreter','latex')


subplot(2,1,2)
hold on
plot(t_all(1:T), x(2,1:T))
plot(t_all(1:T), x_star1(2,1:T))
plot(t_all(1:T), x_star2(2,1:T))
plot(t_all(1:T), x_bar(2,1:T))
xlim([0,1])
ylim([-30,30])
legend('x','x star1','x star2','x bar')
xlabel('time')
ylabel('value')
title('x1 vs t,$\epsilon = 0$, mean = $\frac{\pi}{2}$, $\sigma = \frac{\pi}{8}$','Interpreter','latex')



