%%% The main function for implementing parameter estimation from the article
%%% A Musically Motivated Mid-Level Representation by Jean-Louis Durrieu, Associate Member, IEEE, 
%%% Bertrand David, Member, IEEE, and
%%% Ga?l Richard, Senior Member, IEEE
clear all; close all;
%% Load Music Signal
[x,fs] = audioread('bearlin.wav');
x = x(:,1);
% x = mean(x,2);
% x = x(1:500000);
% L = length(x);
%% Pre-processing
nfft = 2048;
lengthWindow = 2048;
% fs = 44100;
hopsize = 256;
noverlap = lengthWindow - hopsize;
X = stft(x,nfft,hopsize,0,'hann');
SX = abs(X).^2;
[F,N] = size(SX);


%% Generating WF0
minF0 = 100;
maxF0 = 800;
chirpPerF0 = 1;
stepNotes = 20;
Oq = 0.25;
perF0 = chirpPerF0;
% depthChirpInSemiTone = 0.15; 
[F0Table, myWF0] = generateWF0_chirped(minF0,maxF0,fs,nfft,stepNotes,lengthWindow,Oq,perF0); 
% myWF0 = myWF0./(repmat(max(myWF0),size(myWF0,1),1));
myWF0 = myWF0(1:F,:);
myWF0 = myWF0./(repmat(max(myWF0),F,1));

%% Generating WGAMMA
numberFrequencyBins = F;
frequencyScale = 'linear';
numberOfBasis = 30;
overlap = 0.75;
WGAMMA = generateWGAMMA(numberFrequencyBins,numberOfBasis,overlap);
% WGAMMA = WGAMMA./(repmat(max(WGAMMA),F,1));
    
%% Parameter Estimation
load('WF0.mat');
K = 50; R = 40;
iter = 50;
omega = 1;

% [SXhat, HGAMMA, HPHI, HF0, WM, HM, recoError] = SIMM(SX, WF0, WGAMMA, K, R, iter, omega);
SX = max(SX,eps);
[F,N] = size(SX);
NF0 = size(WF0,2);
P = size(WGAMMA,2);
recoError = zeros(iter,5);

%%%% Random Initialization %%%%
HGAMMA = abs(randn(P,K));
HPHI = abs(randn(K,N));
HF0 = abs(randn(NF0,N));
WM = abs(randn(F,R));
HM = abs(randn(R,N));

WPHI = WGAMMA * HGAMMA;
SF0 = WF0 * HF0;
SPHI = WPHI * HPHI;
SM = WM*HM;
% SXhat = SPHI.*SF0 + SM; 

tempNum = zeros(F,N);
tempDen = zeros(F,N);

%%
clear all��
% iter = 100;
load('TestData');
% SXhat = SPHI.*SF0 + SM; 
%%%% Update Rules %%%%
iter = 100;
[SXhat, HGAMMA, HPHI, HF0, WM, HM, recoError] = SIMM(SX, WF0, WGAMMA, K, R, iter, omega);
% for i = 1:iter
% 
%     i
% %%% updating HF0 %%%
%     tempNum = (SPHI .* SX)./max(SXhat.^2, eps);
%     tempDen = SPHI./max(SXhat, eps);
%     HF0 = HF0.*((WF0'*tempNum)./max((WF0'*tempDen), eps)).^omega;
%     SF0 = max(WF0*HF0,eps);
% %     SPHI = max(WGAMMA*HGAMMA*HPHI,eps);
% %     SM = WM*HM;
%     SXhat = max(SPHI.*SF0 + SM,eps);
%     recoError(i,1) = ISDistortion(SX,SXhat);
%     disp(recoError(i,1));
% %%
% %%% updating HPHI %%%
%     tempNum = (SF0 .* SX)./max(SXhat.^2, eps);
%     tempDen = SF0./max(SXhat, eps);
%     HPHI = HPHI.*((WPHI'*tempNum)./max((WPHI'*tempDen), eps)).^omega;
%     sumHPHI = sum(HPHI,1);
%     HPHI(:,sumHPHI>0) = HPHI(:,sumHPHI>0)./repmat(sumHPHI(sumHPHI>0),K,1);
%     HF0 = HF0.*repmat(sumHPHI,NF0,1);
%     SF0 = max(WF0*HF0,eps);
%     SPHI = max(WPHI*HPHI,eps);
% %     SM = WM*HM;
%     SXhat = max(SPHI.*SF0 + SM,eps);
%     recoError(i,2) = ISDistortion(SX,SXhat);
%     disp(recoError(i,2));
% %%
% %%% updating HM %%%
%     tempNum = SX./max(SXhat.^2, eps);
%     tempDen = 1./max(SXhat, eps);
%     HM = max(HM.*((WM'*tempNum)./max(WM'*tempDen, eps)).^omega, eps);
% %     SF0 = max(WF0*HF0,eps);
% %     SPHI = max(WGAMMA*HGAMMA*HPHI,eps);
%     SM = max(WM*HM, eps);
%     SXhat = max(SPHI.*SF0 + SM, eps);
%     recoError(i,3) = ISDistortion(SX,SXhat);
%     disp(recoError(i,3));
% %%
% %%% updating HGAMMA %%%
%     tempNum = (SF0.*SX)./max(SXhat.^2, eps);
%     tempDen = SF0./max(SXhat, eps);
%     HGAMMA = max(HGAMMA.*((WGAMMA'*(tempNum*HPHI'))./max(WGAMMA'*(tempDen*HPHI'), eps)).*omega, eps);
%     sumHGAMMA = sum(HGAMMA,1);
%     HGAMMA(:,sumHGAMMA>0) = HGAMMA(:,sumHGAMMA>0)./repmat(sumHGAMMA(sumHGAMMA>0),P,1);
%     HPHI = HPHI .* repmat(sumHGAMMA',1,N);
%     sumHPHI = sum(HPHI,1);
%     HPHI(:,sumHPHI>0) = HPHI(:,sumHPHI>0)./repmat(sumHPHI(sumHPHI>0),K,1);
%     HF0 = HF0.*repmat(sumHPHI,NF0,1);
%     
%     
%     WPHI = max(WGAMMA*HGAMMA, eps);
%     SF0 = max(WF0*HF0,eps);
%     SPHI = max(WPHI*HPHI,eps);
% %     SM = WM*HM;
%     SXhat = max(SPHI.*SF0 + SM,eps);
%     recoError(i,4) = ISDistortion(SX,SXhat);
%     disp(recoError(i,4));
% %%
% %%% updating WM %%%
%     tempNum = SX./max(SXhat.^2, eps);
%     tempDen = 1./max(SXhat, eps);
%     WM = max(WM.*((tempNum*HM')./max(tempDen*HM', eps)).^omega, eps);
%     sumWM = sum(WM,1);
%     WM(:,sumWM>0) = WM(:,sumWM>0)./repmat(sumWM(sumWM>0),F,1);
%     HM = HM.*repmat(sumWM',1,N);
% %     SF0 = WF0*HF0;
% %     SPHI = WGAMMA*HGAMMA*HPHI;
%     SM = max(WM*HM,eps);
%     SXhat = max(SPHI.*SF0 + SM,eps);
% 
% 
%     recoError(i,5) = ISDistortion(SX,SXhat);
%     disp(recoError(i,5));
%     
% end




