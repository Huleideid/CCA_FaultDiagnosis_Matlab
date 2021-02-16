%% CCA for Fault Diagnosis
clear; 
close all; 
clc; 
%% 加载数据以及数据预处理
load ('.\data\d00.mat')
load ('.\data\d01_te.mat')
X_train=d00(:,[1:22,42:52]); 
Y_train=d00(:,35); 
X_test=d01te(:,[1:22,42:52]); 
Y_test=d01te(:,35); 
%% 数据标准化
[X_train, x_mean,x_v] = autos(X_train);
[Y_train, y_mean,y_v] = autos(Y_train);
X_test=(X_test-repmat(x_mean,size(X_test,1),1))./(repmat(x_v,size(X_test,1),1));
Y_test=(Y_test-repmat(y_mean,size(X_test,1),1))./(repmat(y_v,size(X_test,1),1));
%% CCA建模
[U, S, V, J,J_res, L,L_res] = cca_fun_static(X_train',Y_train');
%% 控制限计算
for i=1:size(X_train,1) 
    r1_old(i,:)=X_train(i,:)*J*S-Y_train(i,:)*L;
    Q_cca_old(i)=r1_old(i,:)*r1_old(i,:)';
end
u_old=mean(Q_cca_old);
S_old=var(Q_cca_old);
g = S_old/(2*u_old);
h = 2*u_old*u_old/S_old;
alpha = 0.05;%alpha为显著性水平
Q_cca_rd = g*chi2inv(1-alpha,h);
%% 计算监控指标
for i=1:size(X_test,1) 
    r1(i,:)=X_test(i,:)*J*S-Y_test(i,:)*L;
    Q_cca(i)=r1(i,:)*r1(i,:)';
end
%% 监控可视化
figure
set(gcf,'color','white')
plot(Q_cca);
hold on 
plot(ones(1,size(Q_cca,2))*Q_cca_rd,'r--');


