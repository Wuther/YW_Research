function window = sinebell(lengthWindow)
L = lengthWindow;
window = sin(pi * (0:L-1)/L);