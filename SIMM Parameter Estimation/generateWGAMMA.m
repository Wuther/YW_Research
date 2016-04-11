function WGM = generateWGAMMA(numberFrequencyBins,numberOfBasis,overlap)


numberOfWindowsForUnit = ceil(1.0/(1-overlap));
overlap = 1.0 - 1.0/double(numberOfWindowsForUnit);
lengthSineWindow = ceil(numberFrequencyBins/((1.0-overlap)*(numberOfBasis - 1)+1-2*overlap));
mappingFrequency = [1:numberFrequencyBins]';
sizeBigWindow = 2.0 * numberFrequencyBins;

firstWindowCenter = -numberOfWindowsForUnit + 1;
lastWindowCenter = numberOfBasis - numberOfWindowsForUnit + 1;
sineCenters = round((firstWindowCenter:lastWindowCenter-1)*(1-overlap)*double(lengthSineWindow)+lengthSineWindow/2.0);

prototypeSineWindow = hann(lengthSineWindow);
bigWindow = zeros(sizeBigWindow*2,1);
bigWindow((sizeBigWindow-lengthSineWindow/2.0):(sizeBigWindow+lengthSineWindow/2.0)-1,1)=prototypeSineWindow;
WGM = zeros(numberFrequencyBins,numberOfBasis);
for p = 1:numberOfBasis
    WGM(:,p) = bigWindow(int32(mappingFrequency - sineCenters(p) + sizeBigWindow));
end

end