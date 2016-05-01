function [F0Table, WF0] = generateWF0_chirped(minF0,maxF0,fs,nfft,stepNotes,lengthWindow,Oq,perF0)
%%% Generate the basis matrix WF0 %%%

U = ceil(log2(maxF0/minF0) * 12 * stepNotes) + 1;
window = sinebell(lengthWindow);
% F0Table = minF0 * (2.^((0:U-1)/(12*stepNotes)));
wf0elements = U * perF0;
t0 = 0;
F = @(u) minF0 * (2.^((u-1)/(12*stepNotes)));
F0Table = F(1:U);
WF0 = zeros(nfft,wf0elements);
for u = 1:U
    HarmonicMax = floor((fs/2)/F(u));%Partialmax
    h = 1:HarmonicMax;%frequence_numbers
    term = 1i * 2 * pi * h * Oq;%temp_array
%     if perF0 > 1
%         for chirpNumber = 1:perF0
%             F2 = F(u) * (2.^(chirpNumber * depthChirpInSemiTone/(12.0 * (perF0-1))));
%             F1 = 2 * F(u) - F2;
%             Ch = F(u) * (27/4) * (exp(-term) + 2 * (1 + 2*exp(-term))/term - 6*(1-exp(-term))/(term.^2));
%             timeStamps = (0:nfft-1)/fs + t0/F(u);
%             odgd = exp(2*1i*pi*(F1*h'*timeStamps+(F2-F1)*h'*(timeStamps.^2/(2*nfft/fs)))).*repmat(Ch',1,nfft);
%             odgd = sum(odgd,1);
%             WF0(:,(u-1)*perF0+chirpNumber) = abs(fft(real(odgd' .* hanning(lengthWindow)),nfft)).^2;
%         end
%     else
        Ch = F(u) * (27/4) * (exp(-term) + (2 * (1 + 2*exp(-term))./term) - (6*(1-exp(-term))./(term.^2)))./term;%amplitudes
        timeStamps = (0:nfft-1)/fs + t0/F(u);
        tempmat = exp((2*1i*pi*F(u)*h')*timeStamps);
        tempnx = Ch.' * ones(1,nfft);
        odgd = tempmat.*tempnx;        
        odgd = sum(odgd,1);
        odgdspec = fft(real(odgd.'.*hanning(lengthWindow)),nfft);
        WF0(:,u*perF0) = abs(odgdspec).^2;
end
    
end

