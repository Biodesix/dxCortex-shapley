% setables
grp_size = 30; % size of each group in training
test_grp_size = 30; % size of each group in test
LEAVE_IN = 2; % number of features for each mini-classifier to use
NB = 20; % number of bags
random_seed = 3; % random seed

% random number generator
rs = rng(random_seed);

% set up features
mu_uncorr   = [2.5, 2.5]; % two features that are uncorrelated with expression difference of 2.5
mu_corr     = [2.0, 2.0]; % two features that are corrlated with expression difference of 2.0
corr_sample_std = 0.5; % standard deviation of distribution to sample when generating off diagonal elements of correlation matrix 
NF_uninformative = 4; % four uninformative features (expression difference of 0)

% derivables from what we've set, but used downstairs
NF_informative = length(mu_corr) + length(mu_uncorr);
NF = NF_uninformative + NF_informative;

% setables, but by default are derivable from things we've already set
NS_training = 2*grp_size; % the number of training samples
NMCS = NF*(NF+1)*0.5; % number of mini-classifiers
NDOI = 10*floor(NMCS/LEAVE_IN); % number of drop-out iterations

% set up the correlation matrices
lambda = find_parm_random_corr(corr_sample_std); 
corr_mat = [];
if (length(mu_corr) > 1)
    corr_mat = vineBeta(length(mu_corr),lambda);
end
%% 

% generate training and validation sample
[r1,r2,rs] = GenReal_for_SV(grp_size, grp_size, NF, mu_uncorr, mu_corr, corr_mat, corr_mat, rs);
[t1,t2,rs] = GenReal_for_SV(grp_size, grp_size, NF, mu_uncorr, mu_corr, corr_mat, corr_mat, rs);

% turn it into tables and name the feautres
varNames = cell(NF,1);
for i = 1:NF
    if( i <= (NF-NF_informative)) % uninformative
        varNames{i} = ['U' num2str(i)];
    elseif ( i > NF - length(mu_corr)) % informative correlated
        varNames{i} = ['IC' num2str(i)];
    else
        varNames{i} = ['IU' num2str(i)]; % informative uncorrelated
    end
end
R = array2table( [r1 ; r2],'VariableNames',varNames);
T = array2table( [t1 ; t2],'VariableNames',varNames);

% make training labels and test labels
trainingLabels(1:grp_size) = 1;
trainingLabels(grp_size+1:NS_training) = 0;
testLabels(1:test_grp_size) = 1;
testLabels((test_grp_size+1): (2*test_grp_size)) = 0;
ftr_table = addvars(R,trainingLabels','NewVariableNames','Labels');

% train the dxCortex model
tic;
fullCortex = dxCortex_pairs(NB,rs,NDOI,LEAVE_IN,7); % NOTE pick different RS for different mus !!!!
fullCortex.train(ftr_table,0.6666666);
train_time = toc

% get prediction on test set
labels = fullCortex.predict(T);
test_accuracy = 1 - sum ( (labels > 0.5) ~= testLabels' )  /size(T,1);
display([ 'test-accuracy = ' num2str(test_accuracy) ])

% get all four types of Shapley values from the paper
% tic;
% [SVs_restricted,~,~] = fullCortex.SVs_restricted(T);
% restricted_time = toc
% 
% tic;
% [SVs_shallow,~,~] = fullCortex.SVs_shallow(T);
% shallow_time = toc
% 
% tic;
% [SVs_deep,~,~] = fullCortex.SVs_deep(T);
% deep_time = toc
% 
tic;
[SVs_hierarchical,~,~] = fullCortex.SVs_hierarchical(T);
hier_time = toc

% plot the Shapley values like in figure 2 from the paper
SVS_r = [SVs_restricted(:,1:grp_size) -1*SVs_restricted(:,grp_size+1:end)];
SVS_s = [SVs_shallow(:,1:grp_size) -1*SVs_shallow(:,grp_size+1:end)];
SVS_d = [SVs_deep(:,1:grp_size) -1*SVs_deep(:,grp_size+1:end)];
SVS_h = [SVs_hierarchical(:,1:grp_size) -1*SVs_hierarchical(:,grp_size+1:end)];

ff = 1./sqrt(size(SVS_d,2));
d_vals = mean(SVS_d,2);
d_err = std(SVS_d,1,2)*ff;
h_vals = mean(SVS_h,2);
h_err = std(SVS_h,1,2)*ff;
s_vals = mean(SVS_s,2);
s_err = std(SVS_s,1,2)*ff;
r_vals = mean(SVS_r,2);
r_err = std(SVS_r,1,2)*ff;

clf("reset")
xs =1:1:length(r_vals);
C = linspecer(4);
hold on
errorbar(xs,r_vals,r_err,'o','Color',C(1,:),"MarkerFaceColor",C(1,:),'DisplayName','restricted')
errorbar(xs,s_vals,s_err,'o','Color',C(2,:),"MarkerFaceColor",C(2,:),'DisplayName','shallow')
errorbar(xs,d_vals,d_err,'o','Color',C(3,:),"MarkerFaceColor",C(3,:),'DisplayName','deep')
errorbar(xs,h_vals,h_err,'o','Color',C(4,:),"MarkerFaceColor",C(4,:),'DisplayName','hierarchical')
xticks(xs)
xticklabels(varNames);
xlabel('features')
ylabel('average SV and error')
legend('Location',"northwest")
xlim([0.5 NF+0.5])
ax = gca;
ax.XAxis.TickDirection = 'out';
hold off