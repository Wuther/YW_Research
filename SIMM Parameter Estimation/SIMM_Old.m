function [SXhat, HGAMMA, HPHI, HF0, WM, HM, recoError] = SIMM(SX, WF0, WGAMMA, K, R, iter, Beta)
%%% Parameters Estimation with NMF Using Multiplicative Updating Rules %%%
%%% Inputs:
% SX: FxN The power spectrum of the signal to be estimated.
% WF0: FxNF0 basis matrix containing NF0 source elements.
% WGAMMA: FxP basis matrix of P smooth elemtentary filters
% K: The number of filters to be fitted
% R: The number of spectral shapes for the accompaniment
% iter; Number of Iterations
% Beta: The power in the update Rule, for IS distortion, Beta = 0.
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
recoError = zeros(iter,1);

%%%% Random Initialization %%%%
HGAMMA = abs(randn(P,K));
HPHI = abs(randn(K,N));
HF0 = abs(randn(NF0,N));
WM = abs(randn(F,R));
HM = abs(randn(R,N));

SF0 = WF0*HF0;
SPHI = WGAMMA*HGAMMA*HPHI;
SM = WM*HM;
SXhat = max(SPHI.*SF0 + SM,eps); 

%%%% Update Rules %%%%

for i = 1:iter

    i
%%% updating HF0 %%%
    tempnum = SXhat.^(Beta-2).*SX;
    tempdem = SXhat.^(Beta-1);
    HF0 = HF0.*(WF0'*(SPHI.*tempnum))./(WF0'*(SPHI.*tempdem));
    SF0 = max(WF0*HF0,eps);
%     SPHI = max(WGAMMA*HGAMMA*HPHI,eps);
%     SM = WM*HM;
    SXhat = max(SPHI.*SF0 + SM,eps);
%     recoError(i) = ISDistortion(SXhat,SX)

%%% updating HPHI %%%
    tempnum = SXhat.^(Beta-2).*SX;
    tempdem = SXhat.^(Beta-1);
    HPHI = HPHI.*((WGAMMA*HGAMMA)'*(SF0.*tempnum))./((WGAMMA*HGAMMA)'*(SF0.*tempdem));
    sumHPHI = sum(HPHI,1);
    HPHI(:,sumHPHI>0) = HPHI(:,sumHPHI>0)./repmat(sumHPHI(sumHPHI>0),K,1);
    HF0 = HF0.*repmat(sumHPHI,NF0,1);
    SF0 = max(WF0*HF0,eps);
    SPHI = max(WGAMMA*HGAMMA*HPHI,eps);
%     SM = WM*HM;
    SXhat = max(SPHI.*SF0 + SM,eps);
%     recoError(i) = ISDistortion(SXhat,SX)

%%% updating HM %%%
    tempnum = SXhat.^(Beta-2).*SX;
    tempdem = SXhat.^(Beta-1);
    HM = HM.*(WM'*tempnum)./(WM'*tempdem);
%     SF0 = max(WF0*HF0,eps);
%     SPHI = max(WGAMMA*HGAMMA*HPHI,eps);
    SM = WM*HM;
    SXhat = max(SPHI.*SF0 + SM,eps);
%     recoError(i) = ISDistortion(SXhat,SX)

%%% updating HGAMMA %%%
    tempnum = SXhat.^(Beta-2).*SX;
    tempdem = SXhat.^(Beta-1);
    HGAMMA = HGAMMA.*(WGAMMA'*(SF0.*tempnum)*HPHI')./(WGAMMA'*(SF0.*tempdem)*HPHI');
    sumHGAMMA = sum(HGAMMA,1);
    HGAMMA(:,sumHGAMMA>0) = HGAMMA(:,sumHGAMMA>0)./repmat(sumHGAMMA(sumHGAMMA>0),P,1);
    HPHI = HPHI .* repmat(sumHGAMMA',1,N);
    sumHPHI = sum(HPHI,1);
    HPHI(:,sumHPHI>0) = HPHI(:,sumHPHI>0)./repmat(sumHPHI(sumHPHI>0),K,1);
    HF0 = HF0.*repmat(sumHPHI,NF0,1);
    SF0 = max(WF0*HF0,eps);
    SPHI = max(WGAMMA*HGAMMA*HPHI,eps);
%     SM = WM*HM;
    SXhat = max(SPHI.*SF0 + SM,eps);
%     recoError(i) = ISDistortion(SXhat,SX)

%%% updating WM %%%
    tempnum = SXhat.^(Beta-2).*SX;
    tempdem = SXhat.^(Beta-1);
    WM = WM.*(tempnum*HM')./(tempdem*HM');
    sumWM = sum(WM,1);
    WM(:,sumWM>0) = WM(:,sumWM>0)./repmat(sumWM(sumWM>0),F,1);
    HM = HM.*repmat(sumWM',1,N);
%     SF0 = WF0*HF0;
%     SPHI = WGAMMA*HGAMMA*HPHI;
    SM = max(WM*HM,eps);
    SXhat = max(SPHI.*SF0 + SM,eps);


    recoError(i) = ISDistortion(SXhat,SX);
    
end

end

