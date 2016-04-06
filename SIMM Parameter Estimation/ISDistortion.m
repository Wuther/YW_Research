function IS = ISDistortion(X,Y)
IS = sum(sum(X./Y-log(X./Y)-1));
end