%% Top-down level prediction
% Written by Shunxin Guo 
% 2019-10-24
%% Inputs:
% input_data: training data without labels
% model: 
% tree: the tree hierarchy
% ע�⣺ֻ����ÿһ��Ԥ�����һ����ǩ����Ȼ���м��ҲԤ���һ����ǩ�����ǲ�û����Ϊ���һ��Ԥ���ǩ�ĸ�����Ϣ
% ������Ը������ƣ����԰��м�һ�������ӣ�����ʹ��Ϊ���ˣ�
%% Output

function [predict_label] = FS_topDownSVMLevelPredictionClu(levels,input_data, model, Clutree, feature,numberSel)
    
    [m,~] = size(input_data);
    root = find(Clutree(:,1)==0);     
	for j = 1:m %The number of samples
        %% ����ÿ�㿪ʼ
        for i = 1:levels
            selFeature = feature{i}(1:numberSel);
            [currentNode] = svmpredict(input_data(j,end),input_data(j,selFeature), model{i},'-q');
             if ismember(currentNode,tree_LeafNode(Clutree))
               
                 break;
             end
        end   
       if (currentNode > root)
           currentNode = root-1;
       end
        predict_label(j) = currentNode;  
    end %%endfor    
end
