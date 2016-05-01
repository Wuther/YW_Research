function [SXhat, HGAMMA, HPHI, HF0, WM, HM, recoError] = SIMM(SX, WF0, WGAMMA, K, R, iter, omega)
%%% Parameters Estimation with NMF Using Multiplicative Updating Rules %%%
%%% Inputs:
% SX: FxN The power spectrum of the signal to be estimated.
% WF0: FxNF0 basis matrix containing NF0 source elements.
% WGAMMA: FxP basis matrix of P smooth elemtentary filters
% K: The number of filters to be fitted
% R: The number of spectral shapes for the accompaniment
% iter; Number of Iterations
% omega: updateRulePower
% stepNotes: The number of elements in WF0 per semitone.

%%% Outputs
% HGAMMA: PxK decomposition matrix of WPHI on WGAMMA
% HPHI: KxN amplitude matrix of the filter part of the lead instrument
% HF0: The NF0xN amplitude matrix for the source part of the lead instrument
% WM: The FxR matrix for spectral shapes of the accompaniment
% HM: The RxN amplitude matrix associated with each of the R accompanimsent
% spectral shapes.
% recoError: The reconstruction errors based on IS Distortion
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
SXhat = SPHI.*SF0 + SM; 

tempNum = zeros(F,N);
tempDen = zeros(F,N);

%%%% Update Rules %%%%

for i = 1:iter

    i
%%% updating HF0 %%%
    tempNum = (SPHI .* SX)./max(SXhat.^2, eps);
    tempDen = SPHI./max(SXhat, eps);
    HF0 = HF0.*((WF0'*tempNum)./max((WF0'*tempDen), eps)).^omega;
    SF0 = max(WF0*HF0,eps);
%     SPHI = max(WGAMMA*HGAMMA*HPHI,eps);
%     SM = WM*HM;
    SXhat = max(SPHI.*SF0 + SM,eps);
    recoError(i,1) = ISDistortion(SX,SXhat);
    disp(recoError(i,1));
%%
%%% updating HPHI %%%
    tempNum = (SF0 .* SX)./max(SXhat.^2, eps);
    tempDen = SF0./max(SXhat, eps);
    HPHI = HPHI.*((WPHI'*tempNum)./max((WPHI'*tempDen), eps)).^omega;
    sumHPHI = sum(HPHI,1);
    HPHI(:,sumHPHI>0) = HPHI(:,sumHPHI>0)./repmat(sumHPHI(sumHPHI>0),K,1);
    HF0 = HF0.*repmat(sumHPHI,NF0,1);
    SF0 = max(WF0*HF0,eps);
    SPHI = max(WPHI*HPHI,eps);
%     SM = WM*HM;
    SXhat = max(SPHI.*SF0 + SM,eps);
    recoError(i,2) = ISDistortion(SX,SXhat);
    disp(recoError(i,2));
%%
%%% updating HM %%%
    tempNum = SX./max(SXhat.^2, eps);
    tempDen = 1./max(SXhat, eps);
    HM = max(HM.*((WM'*tempNum)./max(WM'*tempDen, eps)).^omega, eps);
%     SF0 = max(WF0*HF0,eps);
%     SPHI = max(WGAMMA*HGAMMA*HPHI,eps);
    SM = max(WM*HM, eps);
    SXhat = max(SPHI.*SF0 + SM, eps);
    recoError(i,3) = ISDistortion(SX,SXhat);
    disp(recoError(i,3));
%%
%%% updating HGAMMA %%%
    tempNum = (SF0.*SX)./max(SXhat.^2, eps);
    tempDen = SF0./max(SXhat, eps);
    HGAMMA = max(HGAMMA.*((WGAMMA'*(tempNum*HPHI'))./max(WGAMMA'*(tempDen*HPHI'), eps)).*omega, eps);
    sumHGAMMA = sum(HGAMMA,1);
    HGAMMA(:,sumHGAMMA>0) = HGAMMA(:,sumHGAMMA>0)./repmat(sumHGAMMA(sumHGAMMA>0),P,1);
    HPHI = HPHI .* repmat(sumHGAMMA',1,N);
    sumHPHI = sum(HPHI,1);
    HPHI(:,sumHPHI>0) = HPHI(:,sumHPHI>0)./repmat(sumHPHI(sumHPHI>0),K,1);
    HF0 = HF0.*repmat(sumHPHI,NF0,1);
    
    
    WPHI = max(WGAMMA*HGAMMA, eps);
    SF0 = max(WF0*HF0,eps);
    SPHI = max(WPHI*HPHI,eps);
%     SM = WM*HM;
    SXhat = max(SPHI.*SF0 + SM,eps);
    recoError(i,4) = ISDistortion(SX,SXhat);
    disp(recoError(i,4));
%%
%%% updating WM %%%
    tempNum = SX./max(SXhat.^2, eps);
    tempDen = 1./max(SXhat, eps);
    WM = max(WM.*((tempNum*HM')./max(tempDen*HM', eps)).^omega, eps);
    sumWM = sum(WM,1);
    WM(:,sumWM>0) = WM(:,sumWM>0)./repmat(sumWM(sumWM>0),F,1);
    HM = HM.*repmat(sumWM',1,N);
%     SF0 = WF0*HF0;
%     SPHI = WGAMMA*HGAMMA*HPHI;
    SM = max(WM*HM,eps);
    SXhat = max(SPHI.*SF0 + SM,eps);


    recoError(i,5) = ISDistortion(SX,SXhat);
    disp(recoError(i,5));
    
end

end

