%%�б�ǩ�ľ��࣬��һ��ı�ǩ������һ��
clear
clc
load DDTrain.mat %%aaaΪ������
len = length(unique(data_array(:,end)));
for i = 1:len
    fin = find(data_array(:,end) == i);
    data_cell{i} = data_array(fin,1:end-1);
    xlswrite('DDTrain.xlsx',data_cell{i},i);
end
% [m, n] = size(data_array);            
% data_cell = mat2cell(data_array, ones(m,1), ones(n,1));    % ��data�и��m*n��cell����                      % ��ӱ�������

