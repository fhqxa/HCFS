function [ CluTree ] = k_meansNew(dataset)
% ���ܣ�ʵ��K-means�㷨�ľ��๦�ܣ�
% ���룺    data, Ϊһ�� ���� M��N�� ��ʾ������������M��ʾ����M����������N��ʾÿһ��������ά�ȣ�
%           k_value, ��ʾ����������Ŀ��
% �����    output, ��һ�������� M��������ʾÿһ���������ڵ�����ţ�

% ���ߣ� ����壻
% ʱ�䣺 2017��10��14��
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

%�������У����ѡȡK��������Ϊ��ʼ�ľ������ģ�
data_num = size(data, 1);
temp = randperm(data_num, k_value)';     
center = data(temp, :);

%���ڼ�������������
iteration = 0;
while 1
    %�����������������ĵľ��룻
    distance = euclidean_distance(data, center);
    %����������ÿһ�д�С�������� �����Ӧ��indexֵ����ʵ����ֻ��Ҫindex�ĵ�һ�е�ֵ��
    [~, index] = sort(distance, 2, 'ascend');

    %�������γ��µľ������ģ�
    center_new = zeros(k_value, size(data, 2));
    for i = 1:k_value
        data_for_one_class = data(index(:, 1) == i, :);          
        center_new(i,:) = mean(data_for_one_class, 1);    %��Ϊ��ʼ�ľ�������Ϊ�������е�Ԫ�أ����Բ������ĳ������������Ϊ0�������
    end
   
    %����������������۾�һ��������
    iteration = iteration + 1;
    fprintf('���е�������Ϊ��%d\n', iteration);
    
    % ��������εľ������Ĳ��䣬��ֹͣ����������ѭ����
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

