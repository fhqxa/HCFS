function [X] = proximalL12norm(D, tau)
% min_X 0.5*||X - D||_F^2 + tau*||X||_{1,2}
% where ||X||_{1,2} = sum_i||X^i||_2, where X^i denotes the i-th row of X
X = repmat(max(0, 1 - tau./sqrt(sum(D.^2,2))),1,size(D,2)).*D;


%对于矩阵A=[a b c d]，1./A=[1/a 1/b 1/c 1/d]，而1/A表示的是A的逆
%如果a、b是矩阵，a./b就是a、b中对应的每个元素相除，得到一个新的矩阵；
%.^2是矩阵中的每个元素都求平方
%max(0,A)当A矩阵元素小于0，用0填充
%repmat (A, m, n)，形成mXn的块矩阵，其中每一个元素以举证A为样本来拷贝