%% Top-down level prediction
% Written by Shunxin Guo 
% 2019-10-24
%% Inputs:
% input_data: training data without labels
% model: 
% tree: the tree hierarchy
% 注意：只是在每一层预测出了一个标签，当然在中间层也预测出一个标签，但是并没有作为最后一层预测标签的辅助信息
% 如果可以更加完善，可以把中间一层进行添加，可以使用为回退；
%% Output

function [predict_label] = FS_topDownSVMLevelPredictionClu(levels,input_data, model, Clutree, feature,numberSel)
    
    [m,~] = size(input_data);
    root = find(Clutree(:,1)==0);     
	for j = 1:m %The number of samples
        %% 按照每层开始
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
