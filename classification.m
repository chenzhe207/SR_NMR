

function accuracy = classification(W, Hlabel,Gamma)


% classify process
errnum = 0;
err = [];
prediction = [];
for featureid=1:size(Gamma,2)
    spcode = Gamma(:,featureid);
    score_est =  W * spcode;
    score_gt = Hlabel(:,featureid);
    [maxv_est, maxind_est] = max(score_est);  % classifying
    [maxv_gt, maxind_gt] = max(score_gt);
    
    if(maxind_est~=maxind_gt)
        errnum = errnum + 1;      
    end
end
accuracy = (size(Gamma,2)-errnum)/size(Gamma,2);