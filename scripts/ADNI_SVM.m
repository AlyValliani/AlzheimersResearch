function ADNI_SVM(num_folds)
    
    if ~exist('num_folds', 'var')
        num_folds = 5;
    end
    
    path = '/sonigroup/fmri/ADNI_registered_and_stripped/';
    listing = dir(path);
    
    X = [];
    y = [];
    
    max_nii = 30;
    num_seen = [0, 0, 0];
    dims = [256, 256, 166];
    fprintf('Loading images');
    for i = 1:length(listing)
        if listing(i).isdir || ~strcmp(listing(i).name(end-3:end), ...
                                       '.nii')
            continue
        end
        
        if strcmp(listing(i).name(3:4), 'CN')
            typ = 1;
        elseif strcmp(listing(i).name(3:4), 'MC')
            typ = 2;
        elseif strcmp(listing(i).name(3:4), 'AD')
            typ = 3;
        else
            continue
        end
        
        fprintf('.');
        
        if num_seen(typ) >= max_nii
            continue
        end
        
        imageName = strcat(path, listing(i).name);
        I_t1uncompress = wfu_uncompress_nifti(imageName);
        I_uncompt1 = spm_vol(I_t1uncompress);
        new_X = spm_read_vols(I_uncompt1);
        
        if ~all(size(new_X) == dims)
            continue
        end
            
        num_seen(typ) = num_seen(typ) + 1;
        
        % normalize
        new_X = new_X / max(new_X(:));
        
        X = cat(1, X, new_X(:)');
        y(end+1) = typ;
    end
    y = y';
    fprintf('\n');
    
    disp(size(X));
    disp(size(y));
    
    % shuffle
    sx = size(X);
    order = randperm(sx(1));
    X = X(order, :);
    y = y(order);
    
    %% Perform PCA
    num_pca_wanted = 100;
    [~, ~, v] = svd(X);
    X = v(1:num_pca_wanted, 1:sx(1))';
    
    %% Train SVM
    fprintf('Training\n');
    for fold = 1:num_folds
        fprintf('Fold %d/%d\n', fold, num_folds);
        [train trainID test testID] = ...
            getTrainingAndTesting(X, y, fold, num_folds);
        
        % Trade off commented regions to try to vary parameters for
        % the SVM
        model = svmtrain(trainID, train);
        
        [predictions, acc, probs] = svmpredict(testID, test, ...
                                               model);
        
        makeConfusionMatrix(predictions,testID)
        
        totalAcc(fold) = acc(1);
    end

    fprintf('\n');

    accuracy = mean(totalAcc);
    maxaccuracy = max(totalAcc);
    minaccuracy = min(totalAcc);
    sdaccuracy = std(totalAcc);
    
    fprintf('Accuracy = %f\n',accuracy);
    fprintf('Max Accuracy = %f\n', maxaccuracy);
    fprintf('Min Accuracy = %f\n', minaccuracy);
    fprintf('SD Accuracy = %f\n', sdaccuracy);
    
end

function one_hot = convert_labels(typ)
    one_hot = zeros(3, 1);
    one_hot(typ) = 1;
end

function [train trainID test testID] = ...
        getTrainingAndTesting(brainVectors,brainIDs,fold,num_folds)
    
    numAD = length(find(brainIDs == 1));
    numMCI = length(find(brainIDs == 2));
    numCN = length(find(brainIDs == 3));
    
    fprintf('AD: %d, MCI: %d, CN: %d\n', numAD, numMCI, numCN);
    numBrains = numAD + numMCI + numCN;
    
    %upper and lower fold ratios
    lfr = (fold-1)/num_folds;
    ufr = fold/num_folds;
    
    testInd = zeros(numBrains,1);
    testInd(floor(lfr*numAD+1):floor(ufr*numAD)) = 1;
    testInd(floor(numAD + lfr*numMCI+1):floor(numAD + ufr*numMCI)) = 1;
    testInd(floor(numAD + numMCI + lfr*numCN+1):floor(numAD + numMCI + ...
                                                      ufr*numCN)) = ...
        1;
    
    %make training index as not testing
    trainInd = ~testInd;
    
    %convert the booleans to indices
    trainInd = find(trainInd);
    testInd = find(testInd);

    % get testing
    test = brainVectors(testInd,:);
    testID = brainIDs(testInd);
    
    % get training
    train = brainVectors(trainInd,:);
    trainID = brainIDs(trainInd);
end 

function makeConfusionMatrix(predictions,testID)
    k = length(unique(testID));
    conf = zeros(k);
    for i = 1:k
        for j = 1:k
            conf(i,j) = sum((predictions == i) .* (testID == j));
        end
    end
    fprintf('Confusion matrix:\n')
    disp(conf)
end


