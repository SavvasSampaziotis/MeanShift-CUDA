
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

E = mean((y- data).^2);
E

figure(1); clf; 

subplot(1,2,1);
hold on;
scatter(y(:,1),y(:,2));
scatter(data(:,1),data(:,2));
title(['MeanShift Result Comparison']);
legend('Matlab','CUDA', 'location', 'best');
xlabel('x-axis');
ylabel('y-axis');
% legend(['MSE = ',num2str(E)]);

subplot(1,2,2);
calcTime = 0.575502/tElapsed*100;
calcTimeFROB = 0.575358/tElapsed*100;
MatlabCalcTime = tElapsed/tElapsed*100;
bar([1,2,4],[MatlabCalcTime, calcTime, calcTimeFROB]);
xticks([1,2,4]);
xticklabels({'Matlab','CUDA', 'CUDA-Fast Frob'});
ylabel('xSpeedUp [%]');
title('Time Performance');
legend(['MATLAB: ',num2str(tElapsed),'sec'], 'location', 'best');
suptitle('Meanshift - r15.mat Dataset');














