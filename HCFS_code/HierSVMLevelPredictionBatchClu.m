%% Date: 2018-2-8
% �޸�˵����
% 1. ����������ݼ�ֻѡ10%20%30%40%50%������
% 2. ��������SVM����c=1����
function [accuracyMean, accuracyStd, F_LCAMean, FHMean, TIEmean, TestTime,RealLabel,PredLabel] = HierSVMLevelPredictionBatchClu(data_array, Clutree, feature)
[m, numFeature] = size(data_array);
numFeature = numFeature - 1; 
numFolds  = 10;
k = 1;
baseline = 1;
numFeature = round(numFeature * 0.2);%ֱ��ѡȡ�ٷ�֮20������
% Test all features (baseline)
% Test 50% 40%	30%	20%	10% features.
% for j = 1:9
%     numSeleted = round(numFeature * j * 0.1);
%     accuracyMean(1, k) = numSeleted;
%     accuracyStd(1, k) = numSeleted;
%     F_LCAMean(1, k) = numSeleted;
%     FHMean(1, k) = numSeleted;
%     TIEmean(1, k) = numSeleted;
%     TestTime(1, k) = numSeleted;
%     rand('seed', 1);
%     indices = crossvalind('Kfold', m, numFolds);
%     tic;
%     [accuracyMean(2, k), accuracyStd(2, k), F_LCAMean(2, k), FHMean(2, k), TIEmean(2, k)] = FS_Kflod_TopDownLevelSVMClassifierClu(data_array, numFolds, Clutree, feature, numSeleted, indices);
%     TestTime(2, k) = toc;
%      k = k+1;
% end
 if (baseline == 1)
    accuracyMean(1, k) = numFeature;
    accuracyStd(1, k) = numFeature;
    F_LCAMean(1, k) = numFeature;
    FHMean(1, k) = numFeature;
    TIEmean(1, k) = numFeature;
    TestTime(1, k) = numFeature;
    rand('seed', 1);
    indices = crossvalind('Kfold', m, numFolds);
    tic;
%    [accuracyMean(2, k), accuracyStd(2, k), F_LCAMean(2, k), FHMean(2, k), TIEmean(2, k),RealLabel,PredLabel] = ClassifierClu(data_array, numFolds,Clutree,feature,numFeature, indices);
     [accuracyMean(2, k), accuracyStd(2, k), F_LCAMean(2, k), FHMean(2, k), TIEmean(2, k),RealLabel,PredLabel] = FS_Kflod_TopDownLevelSVMClassifierClu(data_array, numFolds,Clutree,feature,numFeature, indices);
    TestTime(2, k) = toc;
 end
end
