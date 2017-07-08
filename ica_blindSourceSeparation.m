%% This script implements the independent component analysis.
%% It takes a mixture of biomedical signals as input and does
%% blind source separation to separate the individual signals.
%% Independent component analysis eqaution has been solved by 
%% gradient descent optimization algorithm 
%%
clc
clear all;
load('mixture.mat');

% zero mean
Xc = X - repmat(mean(X,2), 1, size(X,2));

%% Decorrelating
[U B V] = svd(Xc);
XcW = U'*Xc;
%% Gradient ascent
[M N] = size(XcW);

XcW*XcW';

W = rand(M);

alpha = 0.005;

a = W*XcW;
for l = 1:100000
    for i = 1:M
        for j = 1:M
            J = zeros(M);
            J(i,j) = 1;
            s2 = M*N*trace(W\J);
            for n = 1: size(XcW, 2)
                if i == 1
                   s1 = -2*tanh(a(i,n))*(XcW(j,n)); % Super Gaussian
                else
                     s1 = (tanh(a(i,n))-a(i,n))*(XcW(j,n)); % Sub Gaussian
                    
                end
            end
            gradAs(i,j) = s1 + s2;
        end
    end
    W = W + alpha*gradAs;
end
 %% Demixing
shat = W*XcW;
plot(shat');
figure,
subplot(2,1,1); plot(shat(1,:));
subplot(2,1,2); plot(shat(2,:));