function [Realization1, Realization2,random_state] =  ...
    GenReal_for_SV(NG1, NG2, NF, mu_uncorr, mu_corr, corr_1, corr_2, seed)
    % GenReal_for_SV  
    %       generates feature tables of NF ftrs for use in dxCortex according to specified expression differences
    %       and correlation matricies
    %
    %       The results are the feature tables for two groups differentiated by
    %       different distribution parameters.
    %       The NF ftrs are generated in 3 groups: 
    %           Uninformative features: 
    %               no expression difference
    %               no correlations
    %               these are the ftrs in positions: 1 to N_UI 
    %                   where N_UI = NF - N_I_UC - N_I_C
    %                   with N_I_UC = length(mu_uncorr) and N_I_C = length(mu_corr)
    %           Informative uncorrelated features:
    %               the expression difference between the groups is in array mu_uncorr
    %               no correlations 
    %               these are the ftrs in positions:  N_UI + 1 to N_UI + N_I_UC
    %           Informative correlated features:
    %               the expression difference between the groups is in array mu_corr
    %               the features are correlated with the correlations passed as input
    %               these are the ftrs in positions:  N_UI + N_I_UC + 1 to NF
    %       H Roder 3 31 2019 copyright Biodesix
    %       update 3/5/2020 by H Roder
    %           fix projection to subtract mu 
    %           introduced a loop over different alphas
    %       update 2/19/2021 by H Roder
    %           modified fore evaluating SVs
    %           cleaned up description and code
    %       updated 2/21/2021 by H Roder 
    %           to have correlation matrix as input
    %           one each for each group
    %       Dependencies: none           
    %
    %   Input:
    %       NG1:                size of grp1
    %       NG2:                size of grp2
    %       NF:                 total number of ftrs
    %       mu_uncorr:          array, expression difference for uncorrelated
    %                           ftrs
    %       mu_corr:            array, expression difference for correlated
    %                           ftrs
    %       corr_1:             matrix of correlations for grp 1
    %       corr_2:             matrix of correlations for grp 2
    %       seed:               random number generator seed
    %       
    %   Output:
    %       
    %       Realization(1/2):   the feature tables for the groups 1/2
    %       random_state:       the state of the random number generator at the
    %                           end
    %       

    % setting the random number generator
    rng(seed);
    
    % feature structure
    N_I_UC = length(mu_uncorr);
    N_I_C = length(mu_corr);
    N_UI = NF - N_I_UC - N_I_C;
    
    % set correlation matrix
    ss1 = eye(NF); 
    ss2 = ss1;
    if (N_I_C >1 )
        idx = NF - N_I_C + 1;
        ss1(idx:NF,idx:NF) = corr_1;
        ss2(idx:NF,idx:NF) = corr_2;
    
        % setting the diagonal back to 1; seems unnecessary but does not
        % hurt
        for i=idx:NF
            ss1(i,i) = 1.0; 
            ss2(i,i) = 1.0; 
        end
    end
    
    % Generate group1 random numbers
    mu1 = zeros(1,NF);    
    r1 = mvnrnd(mu1,ss1,NG1);
    
    % Generate group2 random numbers
    mu2 = [ zeros(1,N_UI) mu_uncorr mu_corr ];
    r2 = mvnrnd(mu2,ss2,NG2);
    
    Realization1 = r1;
    Realization2 = r2;
    
    random_state = rng;
end

