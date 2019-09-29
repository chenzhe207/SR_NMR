%% written by chenzhe 2017.10.27 for Extended_NMR algorithm
clear all
clc
close all

load AR_120_50_40.mat

EachClassNum = 26;
ClassNum = length(unique(sample_label));
p = 50;
q = 40;
occlusion_type = 'sunglasses'; %% occluison type

%% select training and test samples
sample_data = sample_data ./ repmat(sqrt(sum(sample_data .* sample_data)), [size(sample_data, 1) 1]); %normalize
[train_data, train_label, test_data, test_label] = AR_sample_select(sample_data, sample_label, occlusion_type, EachClassNum, ClassNum);
for j = 1 : size(train_data, 2)  %% 计算训练样本的标签矩阵
    a = train_label(j);
    H_train(a,j) = 1;
end
for j = 1 : size(test_data, 2)  %% 计算训练样本的标签矩阵
    b = test_label(j);
    H_test(b,j) = 1;
end
%% Sparse regularized NMR
time = 1;
for lambda = [0:0.01:0.15]
    
[Ztr, Etr] = SR_NMR(train_data, train_data, lambda, p, q);
[Ztt, Ett] = SR_NMR(train_data, test_data, lambda, p, q);
%% Calculate the classification parameter W
W = inv(Ztr * Ztr' + 0.1 * eye(size(Ztr * Ztr'))) * Ztr * H_train';
W = W';
W = normcols(W);
%% classification 
rec(time) = classification(W, H_test, Ztt)
time = time + 1;
end

rec_ar_sunglasses = rec
save rec_ar_sunglasses rec_ar_sunglasses



plot([0:0.01:0.15],rec_ar_sunglasses,'r.-', [0:0.01:0.15], rec_ar_scarf, 'k.-')
subplot(2,1,2);
plot([1:960],Z2(:,1340),'k.-')

% 
imagesc(Z(:,125))
colormap(gray(256))


