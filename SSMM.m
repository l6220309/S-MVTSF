function [y_hat_test] = SSMM(X, y, X_test)
    %% Set the free parameter 
    tau = 0.01;    % parameter for low rank term
    gamma = 0.01; % parameter for sparse term 
    ss = 0.001;    % step size, should be small (theta in paper)


    
    
    %% Train the binary matrix classifier
    %tic;
    [W,b] = SSMM_GFW(X,y,gamma,tau,ss);

    %% Predict the training accuracy
%     sz = size(X);
%     sz_test = size(X_test);

%     X1 = reshape(X,[sz(1)*sz(2),sz(3)]);
%     y_hat = sign(X1'*W1+b);
%     acc = sum(y_hat == y)/length(y);
%     fprintf('Training acc is %.4f\n',acc);

    %% Predict the testing accuracy
    %tic;
    sz = size(X);
    sz_test = size(X_test);
    W1 = reshape(W,[sz(1)*sz(2),1]);
    X_test1 = reshape(X_test,[sz_test(1)*sz_test(2),sz_test(3)]);
    y_hat_test = sign(X_test1'*W1+b);

end

