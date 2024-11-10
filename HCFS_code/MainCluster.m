clear;
clc;
str = {'protein194'};
m = length(str);
% optimization options
rho_1 = 100000;%   rho1: P
rho_2 = 100000; %   rho2: Q
opts.init = 2;      % guess start point from data（数据猜测的开始点）. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-6;   % tolerance. 
opts.maxIter = 9; % maximum iteration number of optimization（优化迭代的最大次数）.
for i =1:m
    filename = [str{i} 'Train'];
    load (filename);
    [CluTree,numClust] = ClusterTreeguo(data_array);%numClust是聚类的数目
%     [CluTree] = HiClusterTree(data_array);
%     [CluTree] = ClusterTreeKmeans(data_array);
    
    [CluX,CluY]=creatSubTablezh(data_array, CluTree);
    [~, numFeature] = size(data_array);
    clear data_array;
    clear tree;
    % Feature selection
    tic;
%% 按照层进行训练W
   levels = max(unique(CluTree(:,2)));
   idx = [];
    for k = 1:levels
            inter = unique(CluTree(find(CluTree(:,2)== k),1));
            XSel = (CluX(inter))';
            YSel = (CluY(inter))';
            [WPQ{k},funcVal,P{k},Q{k}] =  Least_rMTFL(XSel, YSel, rho_1, rho_2, opts);
            [~,feature{k}] = sort(sum(WPQ{k}.*WPQ{k},2),'descend');
%             [~,featureP{k}] = sort(sum(P{k}.*P{k},2),'descend');
%             [~,featureQ{k}] = sort(sum(Q{k}.*Q{k},2),'descend');
            funcValall(:,k) = funcVal;  
    end
 TrainTime =toc;
% flag = 1;
% if (flag == 1)
%     figure;
%     set(gcf,'color','w');
%     plot(sum(funcValall,2) ,'LineWidth',4,'Color',[0 0 1]);
%     set(gca,'FontName','Times New Roman','FontSize',11);
%     xlabel('Iteration number');
%     ylabel('Objective function value');
% end
    %% Test feature batch
    testFile = [str{i}, 'Test.mat'];
    load (testFile);
    clear tree;
    [accuracyMean, accuracyStd, F_LCAMean, FHMean, TIEmean, TestTime,RealLabel,PredLabel] = HierSVMLevelPredictionBatchClu(data_array, CluTree, feature);
%     [accuracyMeanP, accuracyStdP, F_LCAMeanP, FHMeanP, TIEmeanP, TestTimeP] = HierSVMLevelPredictionBatchClu(data_array, CluTree, featureP);
%     [accuracyMeanQ, accuracyStdQ, F_LCAMeanQ, FHMeanQ, TIEmeanQ, TestTimeQ] = HierSVMLevelPredictionBatchClu(data_array, CluTree, featureQ);
      FilePathFull =[ 'F:\All Code\cluster\code_result']; %
      if (~exist(FilePathFull))
        mkdir(FilePathFull);
    end
        filename = [ str{i},'-' num2str(numClust) '-Clu_new.mat'];
        fullFileName = fullfile(FilePathFull, filename);
    save(fullFileName);
%     if (~exist(FilePathFull))
%         mkdir(FilePathFull);
%     end
%         filename = [ str{i} 'Clu3PQ.mat'];
%         fullFileName = fullfile(FilePathFull, filename);
%     save(fullFileName, 'rho_1','rho_2','WPQ','opts', 'accuracyMean', 'accuracyStd', 'F_LCAMean', 'FHMean', 'TIEmean', 'TrainTime', 'feature', 'TestTime');
end                                  