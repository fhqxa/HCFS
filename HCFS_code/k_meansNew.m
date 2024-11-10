function [ CluTree ] = k_meansNew(dataset)
% 功能：实现K-means算法的聚类功能；
% 输入：    data, 为一个 矩阵 M×N， 表示样本集，其中M表示共有M个样本，　N表示每一个样本的维度；
%           k_value, 表示聚类的类别数目；
% 输出：    output, 是一个列向量 M×１，表示每一个样本属于的类别编号；

% 作者： 殷和义；
% 时间： 2017年10月14日
% [m,n] = size(dataset);
len = length(unique(dataset(:,end)));
CluTree = [];
k_value = 20;
for i = 1:len
    fin = find(dataset(:,end) == i);
    data_cell{i} = dataset(fin,1:end-1);
    su(:,i) = sum(data_cell{i});
    data_cellLable{i} = dataset(fin,end);
end
data = su';

%从样本中，随机选取K个样本作为初始的聚类中心；
data_num = size(data, 1);
temp = randperm(data_num, k_value)';     
center = data(temp, :);

%用于计数迭代次数：
iteration = 0;
while 1
    %获得样本集与聚类中心的距离；
    distance = euclidean_distance(data, center);
    %将距离矩阵的每一行从小到大排序， 获得相应的index值，其实我们只需要index的第一列的值；
    [~, index] = sort(distance, 2, 'ascend');

    %接下来形成新的聚类中心；
    center_new = zeros(k_value, size(data, 2));
    for i = 1:k_value
        data_for_one_class = data(index(:, 1) == i, :);          
        center_new(i,:) = mean(data_for_one_class, 1);    %因为初始的聚类中心为样本集中的元素，所以不会出现某类别的样本个数为0的情况；
    end
   
    %输出迭代次数，给眼睛一个反馈；
    iteration = iteration + 1;
    fprintf('进行迭代次数为：%d\n', iteration);
    
    % 如果这两次的聚类中心不变，则停止迭代，跳出循环；
    if center_new == center
        break;
    end
    
    center = center_new;
end

out = index(:, 1);
for u = 1:length(out)
    CluTree = [CluTree;len+out(u),2];
end 
root = length(CluTree);
for j = 1:k_value
CluTree = [CluTree;root+k_value+1,1];
end
CluTree = [CluTree;0,0];
end    

