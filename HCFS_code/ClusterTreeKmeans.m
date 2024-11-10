function [CluTree,k_value] = ClusterTreeKmeans(dataset)
[m,n] = size(dataset);
len = length(unique(dataset(:,end)));
CluTree = [];
for i = 1:len
    fin = find(dataset(:,end) == i);
    data_cell{i} = dataset(fin,1:end-1);
    su(:,i) = sum(data_cell{i});
    data_cellLable{i} = dataset(fin,end);
end
data = su';
k_value = 3;
out = k_means(data, k_value);
for u = 1:length(out)
    CluTree = [CluTree;len+out(u),2];
end 
root = length(CluTree);
for j = 1:k_value
CluTree = [CluTree;root+k_value+1,1];
end
CluTree = [CluTree;0,0];
end