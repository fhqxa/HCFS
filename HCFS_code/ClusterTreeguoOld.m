%% 读取数据
function [Clu,CluLabel,CluTree] = ClusterTreeguoOld(dataset)
[m,n] = size(dataset);
len = length(unique(dataset(:,end)));
CluTree = [];
for i = 1:len
    fin = find(dataset(:,end) == i);
    data_cell{i} = dataset(fin,1:end-1);
    su(:,i) = sum(data_cell{i});
    data_cellLable{i} = dataset(fin,end);
end
pre_data = su';
% for k = 1:len
%     x = data_cell{k}';
%     n = numel(x);
%     x = reshape(x,1,n); 
%     pre_data = [pre_data;x]
% end
%% 数据归一化
% pre_data = data_cell';
[rn,cn] = size(pre_data);
step1_data = zeros(rn,cn);
   for k=1:cn
       %基于均值方差的离群点数据归一化
       xm=mean(pre_data(:,k));
       xs=std(pre_data(:,k));
       for j=1:rn
           if (pre_data(j,k))>xm+2*xs
               step1_data(j,k)=1;
           elseif (pre_data(j,k))<xm-2*xs
               step1_data(j,k)=0;
           else
               step1_data(j,k)=(pre_data(j,k)-(xm-2*xs))/(4*xs);
           end
       end
   end
%    xlswrite('norm_data.xlsx',step1_data,i);
step1_data(find(isnan(step1_data)==1)) = 0
%% 层次聚类
numClust = 4;
% dist_h = 'seuclidean';
% link = 'centroid';
dist_h='spearman';
link='weighted';
hidx = clusterdata(step1_data,'maxclust',numClust,'distance',dist_h,'linkage',link);
for j = 1:numClust
    data = data_cell(find(hidx == j));
    label = data_cellLable(find(hidx == j));
    Clu{len+j} = [];
    CluLabel{len+j} =[];
    for t = 1:length(data)
     Clu{len+j}  = [Clu{len+j};cell2mat(data(t))];
     CluLabel{len+j}  = [CluLabel{len+j};cell2mat(label(t))];
     CluTree = [CluTree;len+j,2];
    end
end
root = length(CluTree);
Clu{len+numClust+1} = dataset(:,1:end-1);
CluLabel{len+numClust+1} = dataset(:,end);
for j = 1:numClust
CluTree = [CluTree;root+numClust+1,1];
end
CluTree = [CluTree;0,0];
end

