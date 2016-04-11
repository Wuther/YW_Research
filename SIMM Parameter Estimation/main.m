%%% The main function for implementing parameter estimation from the article
%%% A Musically Motivated Mid-Level Representation by Jean-Louis Durrieu, Associate Member, IEEE, 
%%% Bertrand David, Member, IEEE, and
%%% Ga?l Richard, Senior Member, IEEE
clear all; close all;
%% Load Music Signal
x = audioread('bearlin.wav');
x = x(:,1);
% L = length(x);
%% Pre-processing
nfft = 2048;
lengthWindow = 2048;
fs = 44100;
hopsize = 256;
noverlap = lengthWindow - hopsize;
X = spectrogram(x,lengthWindow,noverlap,nfft,fs,'yaxis');
SX = abs(X).^2;
[F,N] = size(SX);


%% Generating the fixed dictionaries
minF0 = 100;
maxF0 = 800;
chirpPerF0 = 1;
stepNotes = 20;
Oq = 0.25;
perF0 = chirpPerF0;
depthChirpInSemiTone = 0.15; 
[F0Table, WF0] = generateWF0_chirped(minF0,maxF0,fs,nfft,stepNotes,lengthWindow,Oq,perF0,depthChirpInSemiTone); 
WF0 = WF0(1:F,:);
WF0 = WF0./(repmat(max(WF0),size(WF0,1),1));
%%
numberFrequencyBins = F;
frequencyScale = 'linear';
numberOfBasis = 30;
overlap = 0.75;
WGAMMA = generateWGAMMA(numberFrequencyBins,numberOfBasis,overlap);
% WGAMMA = WGAMMA./(repmat(max(WGAMMA),F,1));
    
%% Parameter Estimation
K = 10; R = 40;
iter = 100;
Beta = 0;

[SXhat, HGAMMA, HPHI, HF0, WM, HM, recoError] = SIMM(SX, WF0, WGAMMA, K, R, iter, Beta);

