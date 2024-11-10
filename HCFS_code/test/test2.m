%% 读取数据
clc;clear all,close all
load DDTrain.mat %%aaa为变量名
len = length(unique(tree(:,1)))-2;
pre_data = data_array(:,1:end-1);
% pre_data = xlsread('DDTrain.xlsx','Sheet1','1:3020');
%% 数据归一化
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
   xlswrite('norm_data.xlsx',step1_data);
%% 层次聚类
numClust = len;
dist_h = 'spearman';
link = 'weighted';
hidx = clusterdata(step1_data,'maxclust',numClust,'distance',dist_h,'linkage',link);
for i = 1:len
    fin = find(hidx == i);
    data_cell{i} = pre_data(fin,1:end);
    data_cell{i}(:,end+1) = 1
    xlswrite('DDTrain.xlsx',data_cell{i},i);
end
data_cell{len+1} = data_array; 
 FilePathFull =[ 'F:\All Code\PQFeature\result']; %
    if (~exist(FilePathFull))
        mkdir(FilePathFull);
    end
        filename = 'DDCluster.mat';
        fullFileName = fullfile(FilePathFull, filename);
    save(fullFileName, 'data_cell');