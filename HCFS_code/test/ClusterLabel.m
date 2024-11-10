%%有标签的聚类，把一类的标签聚类在一起
clear
clc
load DDTrain.mat %%aaa为变量名
len = length(unique(data_array(:,end)));
for i = 1:len
    fin = find(data_array(:,end) == i);
    data_cell{i} = data_array(fin,1:end-1);
    xlswrite('DDTrain.xlsx',data_cell{i},i);
end
% [m, n] = size(data_array);            
% data_cell = mat2cell(data_array, ones(m,1), ones(n,1));    % 将data切割成m*n的cell矩阵                      % 添加变量名称

