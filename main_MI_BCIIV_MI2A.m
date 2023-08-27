clc
clear
All = [];
addpath(genpath('.\'))

for k = 1:9
    str1 =  sprintf("./BCI4_2A/A%d.mat",k);
    load(str1);
    train_x = x;
    train_y = y;
    str2 =  sprintf("./BCI4_2A/B%d.mat",k);
    load(str2);
    test_x = x;
    test_y = y;
    x = cat(3, train_x, test_x);
    y = cat(1,train_y, test_y);

    cv = 10;
    trial = size(y,1);
    num = floor(trial / cv);
    tic
    for m = 1:cv

        train_x = x;
        train_y = y;
        test_x = x(:,:,1+(m-1)*num:m*num);
        test_y = y(1+(m-1)*num:m*num);
        train_x(:,:,1+(m-1)*num:m*num)=[];
        train_y(1+(m-1)*num:m*num)=[];
        
        train_y = train_y + 1;
        test_y = test_y + 1;
        
        %% Filter bank CSP %%
        train_f = []; test_f = []; train_f3 = []; test_f3= [];
        for k = [8 12 16 20 24 26]
            b = fir1(3, [k k+4]/250);

            for i=1:size(train_x,3)
                tmp = squeeze(train_x(:,:,i));
                f_tmp = filter(b, 1, tmp');
                 Xs(:,:,i) = f_tmp';
            end
            for i=1:size(test_x,3)
                tmp = squeeze(test_x(:,:,i));
                f_tmp = filter(b, 1, tmp');
                 Xt(:,:,i) = f_tmp';
            end
                nfilter = 7; gamma = 0.001;
%                 [fTrain1,fTest1] = CSPfeature_s(Xs, train_y, Xt, nfilter);
                [fTrain1_1,fTest1_1] = RCSPfeature_s(Xs(:,1:500,:), train_y, Xt(:,1:500,:), nfilter, gamma);
                [fTrain1_2,fTest1_2] = RCSPfeature_s(Xs(:,251:750,:), train_y, Xt(:,251:750,:), nfilter, gamma);
                [fTrain1_3,fTest1_3] = RCSPfeature_s(Xs(:,501:1000,:), train_y, Xt(:,501:1000,:), nfilter, gamma);
%                 train_f = [train_f, fTrain1];
%                 test_f = [test_f, fTest1];
                Channel = 1:22;
                [fTrain2_1,fTest2_1] = TSfeature_Cov(Xs(Channel,1:500,:), Xt(Channel,1:500,:));
                [fTrain2_2,fTest2_2] = TSfeature_Cov(Xs(Channel,251:750,:), Xt(Channel,251:750,:));
                [fTrain2_3,fTest2_3] = TSfeature_Cov(Xs(Channel,501:1000,:), Xt(Channel,501:1000,:));
                
                fTrain2_1 = [fTrain2_1', fTrain1_1];
                fTest2_1 = [fTest2_1', fTest1_1];
                
                fTrain2_2 = [fTrain2_2', fTrain1_2];
                fTest2_2 = [fTest2_2', fTest1_2];
                
                fTrain2_3 = [fTrain2_3', fTrain1_3];
                fTest2_3 = [fTest2_3', fTest1_3];
                
%                 para.alpha=10; para.beta=1; para.dim = 262;
%                 [fTrain2_1, fTest2_1] = GRRO_FS(fTrain2_1, train_y, fTest2_1, para);
%                 [fTrain2_2, fTest2_2] = GRRO_FS(fTrain2_2, train_y, fTest2_2, para);
%                 [fTrain2_3, fTest2_3] = GRRO_FS(fTrain2_3, train_y, fTest2_3, para);

                para.alpha = 0.1; para.beta = 1; para.gamma = 10; para.k = 1; para.dim = 262;
                [fTrain2_1, fTest2_1] = MDFS_FS(fTrain2_1, train_y, fTest2_1, para);
                [fTrain2_2, fTest2_2] = MDFS_FS(fTrain2_2, train_y, fTest2_2, para);
                [fTrain2_3, fTest2_3] = MDFS_FS(fTrain2_3, train_y, fTest2_3, para);

%                 para.alpha = 0.1; para.beta = 1; para.gamma = 10; para.lamda = 0.4; para.rho = 10; para.c = 2;
%                 para.maxIter = 200; para.minimumLossMargin = 0.001; para.dim = 262;
%                 [fTrain2_1, fTest2_1] = GLFS_FS(fTrain2_1, train_y, fTest2_1, para);
%                 [fTrain2_2, fTest2_2] = GLFS_FS(fTrain2_2, train_y, fTest2_2, para);
%                 [fTrain2_3, fTest2_3] = GLFS_FS(fTrain2_3, train_y, fTest2_3, para);
                
                train_f3 = cat(3, train_f3, fTrain2_1);
                test_f3 = cat(3, test_f3, fTest2_1);
                
                train_f3 = cat(3, train_f3, fTrain2_2);
                test_f3 = cat(3, test_f3, fTest2_2);
                
                train_f3 = cat(3, train_f3, fTrain2_3);
                test_f3 = cat(3, test_f3, fTest2_3);
        end
%         LDA = fitcdiscr(train_f,train_y);
%         yPred=predict(LDA,test_f);
%         acc(m) = 100*mean(test_y==yPred);


        train_f3 = permute(train_f3, [3 2 1]);
        test_f3 = permute(test_f3, [3 2 1]);
        for i=1:length(train_y)
            if train_y(i)==1
               Y_train(i)= -1; 
            end
            if train_y(i)==2
                Y_train(i)=1;
            end
        end

        for i=1:length(test_y)
            if test_y(i)==1
               Y_test(i)= -1; 
            end
            if test_y(i)==2
                Y_test(i)=1;
            end
        end

        yPred = SSMM(train_f3, Y_train', test_f3);
%         yPred = RSMM(train_f3, Y_train', test_f3);
        acc(m) = 100*mean(Y_test==yPred');

        %% CSP %%
%         nfilter = 11; 
%         [Xs,Xt] = CSPfeature_s(train_x,train_y,test_x, nfilter);
        
%         nfilter = 11; gamma = 0.001;
%         [Xs,Xt] = RCSPfeature_s(train_x,train_y,test_x, nfilter, gamma);
%         train_x = Xs;
%         test_x = Xt;

%         para.alpha=100; para.beta=0.1; para.dim = 20;
%         [train_x, test_x] = GRRO_FS(Xs, train_y, Xt, para);  

%         para.alpha = 1; para.beta = 1; para.gamma = 100; para.k = 1; para.dim = 20;
%         [train_x, test_x] = MDFS_FS(Xs, train_y, Xt, para);
   
%         LDA = fitcdiscr(train_x,train_y);
%         yPred=predict(LDA,test_x);
%         acc(m) = 100*mean(test_y==yPred);

        %% TS_Cov %%
%         [Xs,Xt] = TSfeature_Cov(train_x, test_x);
%         train_x = Xs;
%         test_x = Xt;
        
%         para.alpha=100; para.beta=0.1; para.dim = 200;
%         [train_x, test_x] = GRRO_FS(Xs', train_y, Xt', para);  
%         train_x = train_x'; test_x = test_x'; 

%         para.alpha = 1; para.beta = 1; para.gamma = 100; para.k = 1; para.dim = 200;
%         [train_x, test_x] = MDFS_FS(Xs', train_y, Xt', para);
%         train_x = train_x'; test_x = test_x'; 
%         
%         yPred = slda(test_x,train_x,train_y);
%         acc(m) = 100*mean(test_y==yPred);
        
    end
    toc
    All = [All,mean(acc)];

end
mean(All)