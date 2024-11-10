clear
clc
load DDTrain.mat %%aaa为变量名
data = data_array(:,1:end-1);
[m, n] = size(data);            
data_cell = mat2cell(data, ones(m,1), ones(n,1));    % 将data切割成m*n的cell矩阵                      % 添加变量名称
xlswrite('DDTrain.xlsx',data_cell);
