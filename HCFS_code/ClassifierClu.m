%% 10-fold Hierarchical SVM
%% Written by Hong Zhao
% Modified by Shunxin Guo
% 2017-4-11
% 2019-10-24
% Modified by Hong Zhao on May 16th, 2017.
%% Last modified by Shunxin Guo with hierarchical SVM on 2020-12-8.
%% Input
% data: the dataset with feature and label;
% numFolds: 10-fold or 5-fold;
% tree: the hierarchical structure of classes;
% flag: the classficiation with different features is used when flag=1;
% feature: the feature subset for each node;
% numberSel: the number of selected feature;
% indices: it depends on numFolds.
%% Output
function [accuracyMean,accuracyStd,F_LCAMean,FHMean,TIEmean,RealLabel,PredLabel] = ClassifierClu(data, numFolds,Clutree,feature,numberSel,indices)
[M,N]=size(data);
accuracy_k = zeros(1,numFolds);
rand('seed',1);
for k = 1:numFolds
    testID = (indices == k);%//获得test集元素在数据集中对应的单元编号
    trainID = ~testID;%//train集元素的编号为非test元素的编号
    test_data = data(testID,:);
    test_label = data(testID,N);
    train_data = data(trainID,:);
    RealLabel{k} = test_label;
    %% Creat sub table
    [trainDataMod, trainLabelMod] = creatSubTablezh(train_data, Clutree);
%         [trainDataMod, trainLabelMod] = creatSubTablezh(train_data, tree);

%      [trainDataMod,trainLabelMod,tree] = ClusterTreeguo(train_data);
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train classifiers of all internal levels    在每层的每个节点训练一个model
    levels = max(unique(Clutree(:,2)));
    for i = 1:levels
            c = unique(Clutree(find(Clutree(:,2)== i),1))
            trainLabel = [];
            trainData = [];      
                for t = 1:length(c)
                   trainLabel = [trainLabel;trainLabelMod{c(t)}];
                   trainData = [trainData;trainDataMod{c(t)}]; 
                end 

            selFeature = feature{i}(1:numberSel);
             for j = length(unique(trainLabel))
            [modelSVM{i}]  = svmtrain(trainLabel, trainData(:,selFeature), '-c 1 -t 0 -q');
%            [modelSVM{i}] = train(double(sparse(trainLabelMod(unique(Clutree(find(Clutree(:,2)==i))), sparse(sparse(trainDataMod{i})), '-c 2 -s 0 -B 1 -q'))));
%             end
            end
    
    %%           Prediction       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [predict_label] = FS_topDownSVMLevelPredictionClu(levels,test_data, modelSVM, Clutree,feature,numberSel) ;%按层进行预测，shunxin
    predict_label = predict_label';
    PredLabel{k} = predict_label;
    %%          Envaluation       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   [PH(k), RH(k), FH(k)] = EvaHier_HierarchicalPrecisionAndRecall(test_label,predict_label',Clutree);
   [P_LCA(k),R_LCA(k),F_LCA(k)] = EvaHier_HierarchicalLCAPrecisionAndRecall(test_label,predict_label',Clutree);
   TIE(k) = EvaHier_TreeInducedError(M,test_label,predict_label',Clutree);
   accuracy_k(k) = EvaHier_HierarchicalAccuracy(test_label,predict_label', Clutree);%王煜
    
end
 accuracyMean = mean(accuracy_k);
 accuracyStd = std(accuracy_k);
F_LCAMean=mean(F_LCA);
FHMean=mean(FH);
TIEmean=mean(TIE);
end