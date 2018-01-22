
clear;

fileID = fopen('../data/mean_shift_output.bin', 'r');

N = fread( fileID, 1, 'int32' );
D = fread( fileID, 1, 'int32' );

data = fread( fileID, N*D, 'float');

data = [data(1:2:end), data(2:2:end)];

fclose(fileID);



load('../data/meanshift_result.mat');
figure(1); clf; hold on;
scatter(y(:,1),y(:,2));
scatter(data(:,1),data(:,2));
hold off;

E = mean((y- data).^2);
E