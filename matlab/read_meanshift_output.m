
clear;

fileID = fopen('../data/mean_shift_output.bin', 'r');

if fileID < 0
    disp('Error opening file');
    return;
end

N = fread( fileID, 1, 'int32' );
D = fread( fileID, 1, 'int32' );

dataTEMP = fread( fileID, N*D, 'float');

% data = [dataTEMP(1:2:end), dataTEMP(2:2:end)];
% data = reshape(data, N, D);
data = zeros(N,D);
for d=1:D
    data(:,d) = dataTEMP(d:D:end)';
end

fclose(fileID);

save('temp.mat', 'data');

if(D~=2)
    return;
end

load('../data/meanshift_result.mat');
figure(1); clf; hold on;
scatter(y(:,1),y(:,2));
scatter(data(:,1),data(:,2));
hold off;

E = mean((y- data).^2);
E