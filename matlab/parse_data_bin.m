
clear;

load('r15.mat');


fileID = fopen('../data/r15.bin', 'w+');

[N,D] = size(X);

% Write Header
fwrite(fileID, N, 'int32');
fwrite(fileID, D, 'int32');

% Write Main Body [ x11,x12, ... x1D, x21 ... x2D ... xN1 ... xND];
tempX = X';
% tempX = [1:(N*D)];
fwrite(fileID, tempX(:), 'float'); % Data are written in column order...

fclose(fileID);