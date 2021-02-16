clc
clear all
close all
%% Load data and rescale to [-1 1]
load('laser_dataset.mat')
data=laserTargets;
data_mat=cell2mat(data);
data_mat_normalized=rescale(data_mat,-1,1);
%% properly separate input and target data 
data_normalized=num2cell(data_mat_normalized);
input=data_normalized(1:end-1);
target=data_normalized(2:end);
%% Training, Validation, and Test Split
x_train=input(1:4000);
y_train=target(1:4000);

x_validation=input(4001:5000);
y_validation=target(4001:5000);

x_test=input(5001:end);
y_test=target(5001:end);
%% chose hyperparameters on validation set
inputDelays=[2,5,7];
hiddenSizes=[20 30 50];
trainFcn='traingdx';  %'traingdx' is chosen over 'traingdm' and 'trainlm'
lr=[0.1, 0.01 0.001];
mu=[0.5, 0.7, 0.9];
epoch=[100, 500, 1000];
err_thr=inf;
for id=1:length(inputDelays)
    for hs=1:length(hiddenSizes)
        for lri=1:length(lr)
            for mui=1:length(mu)
                for epochi=1:length(epoch)
    
                    net = timedelaynet(inputDelays(id),hiddenSizes(hs),trainFcn);
                    net.trainParam.lr=lr(lri)
                    net.trainParam.mc=mu(mui);
                    net.trainParam.epochs=epoch(epochi);
                    [Xs,Xi,Ai,Ts] = preparets(net,x_train,y_train);
                    [net, tr] = train(net,Xs,Ts,'UseParallel','yes');

estimated_validation=net(x_validation);
err = immse(cell2mat(estimated_validation),cell2mat(y_validation))
if err<err_thr
   chosen_id=id;
    chosen_hs=hs;
    chosen_lri=lri;
    chosen_mui=mui;
    chosen_epochi=epochi;
end
err_thr=err;

                end
            end
        end
    end
end

%% Parameters are chosen on Validation set and we Train on the selected model with whole training set
inputDelays=1:7;
hiddenSizes=50;
trainFcn='traingdx';
lr=0.001;
mu=0.9;
epoch=500;
tr_indices = 1:1:5000; %indices used for training
ts_indices = 5001:1:10092; %indices used for assessment

net.divideFcn = 'divideind';
net.divideParam.trainInd = tr_indices;
net.divideParam.testInd = ts_indices;
net = timedelaynet(inputDelays,hiddenSizes,trainFcn);
net.trainParam.lr=lr;
net.trainParam.mc=mu;
net.trainParam.epochs=epoch;
[Xs,Xi,Ai,Ts] = preparets(net,input(1:5000),target(1:5000));
[net, tr] = train(net,Xs,Ts);
% save('IDNN_trainingRecord.mat','tr');
% save('IDNN_net.mat','net');
%% Test the TDNN on test set
estimated_output_test=net(x_test);
IDNN_tsMSE = immse(cell2mat(estimated_output_test),cell2mat(y_test));
% save('IDNN_tsMSE.mat','IDNN_tsMSE');
t_test=5001:10092;
figure;clf;
plot(t_test,cell2mat(estimated_output_test),'r--'); 
hold on
plot(t_test,cell2mat(y_test),'b--'); 
set(gca, 'xlim', [5001 10092]);
legend('Estimated Output','Target')
title('Comparative Plot on Test Set')
% saveas(gcf,'test_target_output.png')
% saveas(gcf,'test_target_output')
%%  Test the TDNN on training set
estimated_output_train=net(input(1:5000));
IDNN_trMSE = immse(cell2mat(estimated_output_train),cell2mat(target(1:5000)));
% save('IDNN_trMSE.mat','IDNN_trMSE');
t_train=1:5000;
figure;clf;
plot(t_train,cell2mat(estimated_output_train),'r--'); 
hold on
plot(t_train,cell2mat(target(1:5000)),'b--'); 
legend('Estimated Output','Target')
title('Comparative Plot on Training Set')
% saveas(gcf,'train_target_output.png')
% saveas(gcf,'train_target_output')
%% Test the TDNN on validation set
estimated_output_validation=net(x_validation);
IDNN_vlMSE = immse(cell2mat(estimated_output_validation),cell2mat(y_validation));
% save('IDNN_vlMSE.mat','IDNN_vlMSE');