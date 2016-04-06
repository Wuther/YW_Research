function WF0 = generateWF0_chirped(minF0,maxF0,fs,nfft,stepNotes,lengthWindow,Oq,perF0,depthChirpInSemiTone)
%%% Generate the basis matrix WF0 %%%

U = ceil(log2(maxF0/minF0) * 12 * stepNotes) + 1;
wf0elements = U * perF0;
t = 0;
F = @(u) 100 * 2^((u-1)/48);
WF0 = zeros(nfft,wf0elements);
for u = 1:U
    HarmonicMax = floor((fs/2)/F(u));
    h = 1:HarmonicMax;
    term = 1i * 2 * pi * h * Oq;
    for chirpNumber = 1:perF0
        F2 = F(u) * (2.^(chirpNumber * depthChirpInSemiTone/(12.0 * (perF0-1))));
        F1 = 2 * F(u) - F2;
        Ch = F(u) * (27/4) * (exp(-term) + 2 * (1 + 2*exp(-term))/term - 6*(1-exp(-term))/(term.^2));
        timeStamps = (0:nfft-1)/fs + t/F(u);
        odgd = exp(2*1i*pi*(F1*h'*timeStamps+(F2-F1)*h'*(timeStamps.^2/(2*nfft/fs)))).*repmat(Ch',1,nfft);
        odgd = sum(odgd,1);
        WF0(:,(u-1)*perF0+chirpNumber) = abs(fft(real(odgd' .* hanning(lengthWindow)),nfft)).^2;
    end
    
end

end