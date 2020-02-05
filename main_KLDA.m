clear
filename = '../wine/wine.txt';
[cls,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13] = textread(filename,'%n%n%n%n%n%n%n%n%n%n%n%n%n%n','delimiter',',');
total = [cls,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13];
%% 样本总数
N = size(total,1);
%% 每类样本数
N1 = sum(total(:,1)==1);
N2 = sum(total(:,1)==2);
N3 = sum(total(:,1)==3);
%% 每类样本
cls1_data = total(1:N1,2:end);
cls2_data = total(N1+1:N1+N2,2:end);
cls3_data = total(N1+N2+1:N,2:end); 
%% 总样本
cls_data = [cls1_data;cls2_data;cls3_data];
%% 数据标准化
cls1_data = zscore(cls1_data);
cls2_data = zscore(cls2_data);
cls3_data = zscore(cls3_data);
cls_data = zscore(cls_data);
E_cls1 = mean(cls1_data);
E_cls2 = mean(cls2_data);
E_cls = mean(cls_data);
% patterns = cls_data;
cov_size = N;
%% B
B1 = 1/N1*ones(N1,N1);
B2 = 1/N2*ones(N2,N2);
B3 = 1/N3*ones(N3,N3);
B = blkdiag(B1,B2,B3);
%% 类间方差矩阵
% Sb = 1/N*cls_data'*B*cls_data;
%% 类内方差矩阵
% Sw = 1/N*(cls_data'*cls_data);

% %% 计算核矩阵
K = cls_data*cls_data';
A = pinv(K)*pinv(K)*K*B*K;
%% 特征值分解
[V,D] = eig(A);
V = real(V);
D = real(D);
% [x,index] = sort(real(diag(evaltures_1)));
% evals=flipud(x);
% index=flipud(index);
%% 将特征向量按特征值的大小顺序排序
eigValue = diag(D);
[Yt, index] = sort(eigValue, 'descend');  
eigVector = V(:,index);
eigValue = eigValue(index);

D = eigValue;
rat1 = D./sum(D);
rat2 = cumsum(D)./sum(D);
%% 调出特征值，贡献率，累计贡献率
result1(1,:)={'特征值','贡献率','累计贡献率'};
result1(2:1+size(total,1),1)=num2cell(D);
result1(2:1+size(total,1),2)=num2cell(rat1);
result1(2:1+size(total,1),3)=num2cell(rat2);
%% 主成分载荷
threshold = 0.85;
index = find(rat2 > threshold);
%% Normalization
norm_eigVector = sqrt(sum(eigVector.^2));
eigVector = eigVector./repmat(norm_eigVector,size(eigVector,1),1);
%% dimension reduction
V = eigVector;
data_klda = K * V(:,1:index(1));
%% 得到新数据
label = total(:,1);
new = [label data_klda];
save('KLDA.mat','data_klda','label')

%% 得到新数据
label = total(:,1);
new = [label data_klda];
figure;
new_1 = data_klda(1:59,:);
new_2 = data_klda(60:130,:);
new_3 = data_klda(131:178,:);
for i=1:size(new_1,1)
    h1 = plot(new_1(i,1),new_1(i,2),'.r');
    hold on;
end
for i=1:size(new_2,1)
    h2 = plot(new_2(i,1),new_2(i,2),'*b');
    hold on;
end
for i=1:size(new_3,1)
    h3 = plot(new_3(i,1),new_3(i,2),'vc');
    hold on;
end
legend([h1(1),h2(1),h3(1)],'class1','class2','class3')
title('KLDA');