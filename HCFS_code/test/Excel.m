clear
clc
load DDTrain.mat %%aaaΪ������
data = data_array(:,1:end-1);
[m, n] = size(data);            
data_cell = mat2cell(data, ones(m,1), ones(n,1));    % ��data�и��m*n��cell����                      % ��ӱ�������
xlswrite('DDTrain.xlsx',data_cell);
