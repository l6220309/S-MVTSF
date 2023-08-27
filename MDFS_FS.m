function [train_x,test_x] = MDFS_FS(Xs, train_y, Xt, para)
    Y_train=zeros(length(train_y), 2);
    for i=1:length(train_y)
        if train_y(i)==1
           Y_train(i,1)=1; 
        end
        if train_y(i)==2
            Y_train(i,2)=1;
        end
    end
    
    [ W, obj ] = MDFS( Xs, Y_train, para );
    [dumb idx] = sort(sum(W.*W,2),'descend'); 
    feature_idx = idx(1:para.dim);
    train_x = Xs(:, feature_idx);
    test_x = Xt(:, feature_idx);
end

