clear all; close all
load census;
% Initialization
% z: The observation
% uPn: Mean of the prior
% vPn: Variance of the prior
% K : Kalman-Gain
% uFn : Mean of the posterior
% vFn : Variance of the posterior
N = length(pop);
alpha = 0.95;
apha = 0.5;
data = pop'/5;
z = data + randn(1,N);
figure(1)
plot(z);
hold on 
plot(data,'r');
x_hat = zeros(1,N);

sigmau = 1;
sigmav = 1;
uPn = zeros(1,N);
uPn_err = zeros(1,N);
vPn = zeros(1,N);
uPn(1) = 0;
vPn(1) = 1;
K = zeros(1,N);
uFn = zeros(1,N);
vFn = zeros(1,N);
uFn_err = zeros(1,N);


% Begin Iteration;

for i = 1:N
    % Correction/Update
    K(i) = vPn(i)/(vPn(i)+sigmav^2);
    uFn(i) = uPn(i) + K(i) * (z(i)-uPn(i));
    uFn_err(i) = uPn_err(i) + K(i) * (z(i)-uPn_err(i));
    vFn(i) = (1-K(i))*vPn(i);
    
    % Estimation
    x_hat(i) = uFn(i);
    
    % Prediction
    if i<N
        uPn(i+1) = alpha * uFn(i);
        uPn_err(i+1) = apha * uFn_err(i);
%         vPn(i+1) = alpha^2 * vFn(i) + sigmau^2;
        vPn(i+1) = alpha^2 * vFn(i) + 1;
    end
end
%     % Correction/Update
%     i = N;
%     K(i) = vPn(i)/(vPn(i)+sigmav^4);
%     uFn(i) = uPn(i) + K(i) * (z(i)-uPn(i));
%     vFn(i) = (1-K(i))*vPn(i);
%     
%     % Estimation
%     x_hat(i) = uFn(i);
%     
    clear i;
    
    figure(2)
    plot(x_hat);
    hold on;
    plot(z,'r.-');
    hold on;
    plot(data,'g--');
    legend('Kalman-Filter','MLE','data');
    title('Kalman-Filter vs MLE');

    errMLE = sum(norm(z-data));
    errKF = sum(norm(x_hat-data));









