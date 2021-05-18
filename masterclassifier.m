classdef masterclassifier < handle
    % class for a single master classifier with methods to train, predict, and generate Shapley values
    
    properties (GetAccess = public, SetAccess = private)
        NF                              % number of features
        K                               % kNN K
        L                               % leave-in number
        Ndoi                            % number of drop-out iterations
        Nmcs                            % number of mini-classifiers
        random_seed                     % random number seed
        features                        % (NF,1) the featurenames
        mc_ftrs                         % (Nmcs,L) the features in a mc (mini-classifier); 
                                        % if there is only one it contains minus 1
        mc_ftr_number                   % (Nmcs,1) contains the # of ftrs used in a mc 
        doi_mcs                         % (Ndoi,L) the mc's in a doi (drop-out iteration)
        log_regs                        % (Ndoi,L+1) the logistic regression coefficients per
                                        %   doi: beta_0 = log_regs(*,1);
                                        %   beta_i = log_regs(*,i);
        ftrs_usedby_doi_and_mc          % (Ndoi,L,2) the ftr used by doi and mc
                                        % if there are less than 2 ftrs in
                                        % a mc it contains -1
        N_ftrs_usedby_doi               % (Ndoi,1) the number of ftrs used by a doi
        ftrs_usedby_doi                 % (Ndoi,2*L) the features used by a doi ordered by
                                        % the mcs in a doi, if there are
                                        % overall less than 2*L the
                                        % remainder are -1 
        doi_ftrSequence                 % (Ndoi,NF) zeros or ones indicating the presence or 
                                        % absence of a feature in the list 1,...,NF                               
        train_data                      % table (NS_train,NF+1) the training ftr table
                                        % the columnh 'Label' contains the
                                        % training labels (0/1)
        number_train_samples            % number of samples in training set  
        mcs                             % (Nmcs,1) array of knn models
    end
    
    methods
        
        function obj = masterclassifier(seed,nDois,nLeavein,k)
            %% set the parameters of the masterclassifier
            obj.random_seed = seed;
            obj.Ndoi = nDois;
            obj.L = nLeavein;
            obj.K = k;
        end
        
        function train(obj,data)
            
            %% setting some overall parameters
            obj.train_data = data;
            obj.NF = size(data,2) - 1;
            obj.features = data.Properties.VariableNames(1:obj.NF);
            obj.number_train_samples = size(data,1);
            obj.Nmcs = obj.NF*(obj.NF+1)/2;
            train_labels = obj.train_data(:,'Labels');
            
            %% setting up mcs
            obj.mc_ftrs = -1*ones(obj.Nmcs,2);
            obj.mc_ftrs(1:obj.NF,1) =  1:1:obj.NF ;
            obj.mc_ftrs( (obj.NF+1):end,:) = nchoosek( 1:1:obj.NF  ,2);
            obj.mc_ftr_number(1:obj.NF) = 1;
            obj.mc_ftr_number( (obj.NF+1):obj.Nmcs) = 2;
            
            %% initializing random numbers
            rng(obj.random_seed);
            
            %% setting up dois and structural information on master classifier
            obj.doi_mcs = randi(obj.Nmcs,obj.Ndoi,obj.L);
            obj.ftrs_usedby_doi_and_mc = -1*ones(obj.Ndoi,obj.L,2);
            obj.ftrs_usedby_doi = -1*ones(obj.Ndoi,2*obj.L);
            obj.N_ftrs_usedby_doi = zeros(obj.Ndoi,1);
            obj.doi_ftrSequence = zeros(obj.Ndoi,obj.NF);
            for i=1:obj.Ndoi
                ic = 0;
                for j=1:obj.L
                    mc_index = obj.doi_mcs(i,j);
                    for k=1:obj.mc_ftr_number(mc_index)
                        obj.ftrs_usedby_doi_and_mc(i,j,k) = ...
                            obj.mc_ftrs(mc_index,k);
                        tmp = obj.mc_ftrs(mc_index,k);
                        ic = ic + 1;
                        obj.ftrs_usedby_doi(i,ic) = tmp;
                        obj.doi_ftrSequence(i,tmp) = 1;
                    end
                end
                obj.N_ftrs_usedby_doi(i) = sum(obj.doi_ftrSequence(i,:));
            end
            
            %% generate mc_labels and store kNNs            
            obj.mcs =  cell(obj.Nmcs,1);
            mc_labels = zeros(obj.number_train_samples,obj.Nmcs);
            for i = 1:obj.Nmcs
                tt = obj.generate_subtable_for_mc(i,obj.train_data);
                mm = fitcknn(tt,train_labels,'NumNeighbors',obj.K);
                obj.mcs{i} = mm; 
                tmp = obj.mcs{i}.predict(tt);
                mc_labels(:,i) = tmp;
            end
            
            %% generate log_reg coefficients
            obj.log_regs = zeros(obj.Ndoi,obj.L + 1);
            warning('off','stats:glmfit:IterationLimit')
            warning('off','stats:LinearModel:RankDefDesignMat')
            warning('off','stats:glmfit:PerfectSeparation')
            for i=1:obj.Ndoi
                tt = mc_labels(:,obj.doi_mcs(i,:));
                log_model = fitglm( ...
                    tt, ...
                    table2array(train_labels),...
                    'Distribution',"binomial",'link','logit');
                coeffs = log_model.Coefficients.Estimate;
                obj.log_regs(i,:) = coeffs;
            end
            warning('on','stats:glmfit:IterationLimit')
            warning('on','stats:LinearModel:RankDefDesignMat')
            warning('on','stats:glmfit:PerfectSeparation')
            
        end
        
        function values = predict(obj,test_data)
            
            %% make the mc_labels for the test_data
            mc_labels = zeros(size(test_data,1),obj.Nmcs);
            for i = 1:obj.Nmcs
                tt = obj.generate_subtable_for_mc(i,test_data);
                tmp = obj.mcs{i}.predict(tt);
                mc_labels(:,i) = tmp;
            end
            
            %% now loop over samples and saverage over dois
            NS = size(test_data,1);
            values = zeros( size(test_data,1),1);
            for j=1:NS
                dd = 0.0;
                for i = 1:obj.Ndoi
                    expo = obj.log_regs(i,1) + ...
                        dot(mc_labels(j,obj.doi_mcs(i,:)),obj.log_regs(i,2:end));
                    dd = dd + 1.0/(1.0+exp( -1.0*expo));
                end
                values(j) = dd/obj.Ndoi;
            end
            
        end
        
        function tt = generate_subtable_for_mc(obj,i,data)
            tt = table();
            for j=1:obj.mc_ftr_number(i)
                ftr2add = obj.mc_ftrs(i,j);
                ftrname = obj.features{ftr2add};
                tt =addvars(tt,data.(ftrname),'NewVariableNames',ftrname);
            end
        end
        
        function mc_labels = generate_mc_labels_for_data(obj,data)
            %% make the mc_labels for the test_data
            % uses pre-stored fitcKnn objects
            mc_labels = zeros(size(data,1),obj.Nmcs);
            for i = 1:obj.Nmcs
                tt = obj.generate_subtable_for_mc(i,data);
                tmp = obj.mcs{i}.predict(tt);
                mc_labels(:,i) = tmp;
            end
        end
        
        function red_labels = reduce_mcLabels_shallow(obj,subset,mc_labels)
            red_labels = zeros(size(mc_labels,1),size(mc_labels,2));
            for i=1:obj.Nmcs
                mcftrs = obj.mc_ftrs(i,1:obj.mc_ftr_number(i));
                tmp = sum ( ismember(mcftrs,subset));
                if ( tmp == obj.mc_ftr_number(i) )
                    red_labels(:,i) = mc_labels(:,i);
                else
                    red_labels(:,i) = 0.5 * ones(size(mc_labels,1),1);
                end
            end
        end
        
        function red_labels = reduce_mcLabels_deep(obj,subset,mc_labels)
            red_labels = zeros(size(mc_labels,1),size(mc_labels,2));
            for i=1:obj.Nmcs
                mcftrs = obj.mc_ftrs(i,1:obj.mc_ftr_number(i));
                tmp = sum ( ismember(mcftrs,subset));
                if ( tmp == obj.mc_ftr_number(i) )
                    red_labels(:,i) = mc_labels(:,i);
                elseif ( tmp == 0 )
                    red_labels(:,i) = 0.5 * ones(size(mc_labels,1),1);
                else
                    ftr = mcftrs(ismember(mcftrs,subset));
                    red_labels(:,i) = mc_labels(:,ftr);
                end
            end
        end
        
        function doi_labels = predict_subset_doi(obj,samples,subset,mc_labels)
            
            NS = size(samples,1);
            
            % make a binary sequence of the subset
            subset_sequence = zeros(1,obj.NF);
            subset_sequence(subset) = 1;
            
            % loop over dois
            ui = zeros(1,NS);
            ic = 0;
            for i=1:obj.Ndoi
                tmp = sum( subset_sequence & obj.doi_ftrSequence(i,:) );
                if ( tmp == obj.N_ftrs_usedby_doi(i) )
                    tmpl = zeros(1,NS);
                    for j=1:NS
                        expo = obj.log_regs(i,1) + ...
                            dot( obj.log_regs(i,2:end) , mc_labels(j, obj.doi_mcs(i,:)) );
                        tmpl(j) = 1.0/(1.0 + exp( -1.0 * expo));
                    end
                    ui = ui + tmpl;
                    ic = ic + 1;
                end
            end
            
            if (ic == 0)
                doi_labels = 0.5*ones(1,NS);
            else
                doi_labels = ui/ic;
            end
        end
        
        function labels = predict_from_reduced_set(obj,redLabels)
            
            NS = size(redLabels,1);
            labels = zeros(NS,1);
            for i= 1:obj.Ndoi
                tmpl = zeros(NS,1);
                for j=1:NS
                    expo = obj.log_regs(i,1) + ...
                        dot( obj.log_regs(i,2:end) , redLabels(j, obj.doi_mcs(i,:)) );
                    tmpl(j) = 1.0/(1.0 + exp( -1.0 * expo));
                end
                labels = labels + tmpl;
            end
            
            labels = labels / obj.Ndoi;
        end
        
        function [SVs,phi_all,phi_0] = SVs_exact_LS(obj, samples, behaviour)

            %   use LS formulation to generate a samples' SV
            %   (Williamson & Feng)
            %   NOTE: this rescales labels to go from -1 to 1 with uninformative being
            %   0

            T = samples;
            NF = size(T,2); 
            NS = size(T,1);
            % initialize RR matrix and S vector
            RR = zeros(NF+1,NF+1);
            S = zeros(NF+1,NS);

            %make an array of ftr indices
            ftrs = 1:1:NF;

            % full mC labels
            mc_labels = obj.generate_mc_labels_for_data(T);

            % generate phi_all
            phi_all = 2*obj.predict(T)-1;

            % generate phi_0 
            ui_labels = 0.5*ones(NS,obj.Nmcs);
            phi_0 = 2*obj.predict_from_reduced_set(ui_labels)-1;
            
            for s = 0:1:NF 
                if (s == 0)
                    tmpRR = zeros(NF+1,NF+1);
                    tmpRR(1,1) = 1;
                    RR = RR + tmpRR;
                    for i = 1:NS
                        S(:,i) = S(:,i) + phi_0(i) * ones(NF+1,1);
                    end
                elseif (s == NF)
                    tmpRR = ones(NF+1,NF+1); 
                    RR = RR + tmpRR;
                    for i=1:NS
                        S(:,i) = S(:,i) + phi_all(i) * ones(NF+1,1);
                    end
                else
                    factor = 1./(nchoosek(NF,s)*s*(NF-s));
                    sizeOfS = nchoosek(NF,s);
                    use = true(NF,1);
                    allSets = nchoosek(ftrs(use),s);
                    for i = 1:sizeOfS 
                        % get a configuration                     
                        currSequence = zeros(NF,1);
                        currSequence(allSets(i,:)) = 1;
                        % make z vector
                        z = [1 currSequence'];
                        % update matrix
                        RR = RR + (z'*z) * factor;
                        % get a classification
                        if (behaviour == emptySetBehavior.doi)
                            u = 2*obj.predict_subset_doi(T,allSets(i,:),mc_labels) - 1.0;
                        elseif (behaviour == emptySetBehavior.mc_shallow)
                            redLabels = obj.reduce_mcLabels_shallow( ...
                                allSets(i,:),mc_labels);
                            u = 2*obj.predict_from_reduced_set(redLabels) - 1.0;
                        elseif (behaviour == emptySetBehavior.mc_deep)
                            redLabels = obj.reduce_mcLabels_deep( ...
                                allSets(i,:),mc_labels);
                            u = 2*obj.predict_from_reduced_set(redLabels) - 1.0;
                        end
                        for jj = 1:NS
                            S(:,jj) = S(:,jj) + z' * factor * u(jj);
                        end
                    end
                end
            end
            SVs = obj.generate_LL_SVs_multiple_samples_spec0_all(RR,S,phi_all,NF,phi_0);
        end
        
        function [SVs_all,phi_all,phi_0] = SVs_reverse_LS(obj,samples, behavior)
            % reverse sum LS for shallow SVs

            NF = length(obj.features);
            NS = size(samples,1);
            % get the mini classifier labels
            mc_labels= obj.generate_mc_labels_for_data(samples);

            % initialize the SVs, phi
            SVs_all = zeros(NF,NS);
            phi_0  = zeros(1,NS);
            phi_all = zeros(1,NS);

            for i = 1:obj.Ndoi
                [SVs, p_all, p_0] = obj.SVs_for_doi_LS(i,behavior,mc_labels);
                SVs_all = SVs_all + SVs;
                phi_0 = phi_0 + p_0;
                phi_all = phi_all + p_all;
            end

            SVs_all = SVs_all/obj.Ndoi;
            phi_0 = phi_0/obj.Ndoi;
            phi_all = phi_all/obj.Ndoi;

        end       
        
        function [SVs,phi_all,phi_0,alphaLogReg] = SVs_from_hierarchy(obj,samples)
            % hierarchical decomposition SVs

            % mc_labels, some constants, initialize SVs
            mc_labels = obj.generate_mc_labels_for_data(samples);       % (NS x Nmcs) mini-classifier labels
            NF = obj.NF;                                                % the number of features    
            NS = size(mc_labels,1);                                     % the number of samples
            Nmcs = obj.Nmcs;                                            % the number of mini-classifiers
            SVs = zeros(NF,NS);

            % mini_classifier SVs and support data
            kNN_SVs = NaN(Nmcs,NS,2);       % set non-existing ones to NaN, can be one or 2 for singles and pairs
            kNN_phi0s = zeros(Nmcs,NS);     % phi_0's for mini-classifiers
            kNN_phialls = zeros(Nmcs,NS);   % phi_all's for mini-classifiers
            kNN_ftrs = zeros(Nmcs,2);       % contains the ftrs used in the mini classifiers
            for i = 1:Nmcs
                [ss,phi0,phiall,ftrs] = obj.SVs_for_a_kNN(i,mc_labels);
                if (length(ftrs) == 1)
                    kNN_SVs(i,:,1) = ss(1,:);
                    kNN_ftrs(i,1) = ftrs(1);
                else
                    kNN_SVs(i,:,1) = ss(1,:);
                    kNN_SVs(i,:,2) = ss(2,:);
                    kNN_ftrs(i,1) = ftrs(1);
                    kNN_ftrs(i,2) = ftrs(2);
                end
                kNN_phi0s(i,:) = phi0;
                kNN_phialls(i,:) = phiall;
            end

            % get RR matrix
            RR = obj.pre_calculate_doi_RR();    % pre calculates the RR matrix (2.2) as it is the same for all dois

            % gather it all up
            phi_all = zeros(1,NS);
            phi_0 = zeros(1,NS);
            for doi = 1:obj.Ndoi       
                % get the SVs for a doi
                [mc_SVs,p_all,p_0,alphaLogReg] = obj.kNN_SVs_for_doi(RR,doi,mc_labels);     % get SVs for mini-classifiers for a doi
                phi_all = phi_all + p_all;
                phi_0 = phi_0 + p_0;
                for imc=1:obj.L
                    mc_index = obj.doi_mcs(doi,imc);
                    NF = obj.mc_ftr_number(mc_index);          
                    for i= 1:NS
                        SVs(kNN_ftrs(mc_index,1),i) = SVs(kNN_ftrs(mc_index,1),i) + ...
                            mc_SVs(imc,i) * kNN_SVs(mc_index,i,1) ... 
                        / ( kNN_phialls(mc_index,i) - kNN_phi0s(mc_index,i));    
                    end
                    if (NF == 2)
                        for i = 1:NS
                            SVs(kNN_ftrs(mc_index,2),i) = SVs(kNN_ftrs(mc_index,2),i) + ...
                                mc_SVs(imc,i) * kNN_SVs(mc_index,i,2) ... 
                            / ( kNN_phialls(mc_index,i) - kNN_phi0s(mc_index,i));    
                        end
                    end
                end        
            end

            SVs = SVs/obj.Ndoi;
            phi_all = phi_all/obj.Ndoi;
            phi_0 = phi_0/obj.Ndoi;

        end
        
        function [SV] = generate_LL_SVs_multiple_samples_spec0_all(obj, RR,S,phi_all,NF,phi_0)
            %   Input
            %       RR: matrix of size NF+1 x NF+1 containing z' W z
            %       S: matrix each column contains z' W v (v is the classification or
            %               utility)
            %       sample_labels: vector of classification labels, length = number of
            %       samples
            %       NF: number of features
            %   Output
            %       SV: shapley values, matrix

            % set up linear system
            X_matrix = zeros(NF+3,NF+3);
            X_matrix(1:(NF+1),1:(NF+1)) = 2*RR;
            G = zeros(2,(NF+1));
            G(2,1:(NF+1)) = 1;
            G(1,1) = 1;
            X_matrix(1:(NF+1), (NF+2):end) = G';
            X_matrix((NF+2):end,1:(NF+1)) = G;

            NS = size(S,2);
            rhs = zeros(NF+3,NS);
            rhs(1:(NF+1),:) = 2*S;
            rhs(NF+2,:) = phi_0;
            rhs(NF+3,:) = phi_all ; % use all features

            % solve the system
            [sol,~] = linsolve(X_matrix,rhs);

            SV = sol(2:(NF+1),:);
        end
        
        function [SVs, phi_all, phi_0] = SVs_for_doi_LS(obj,doi,behavior,mc_labels)
            % LS formalism SVs for a doi

            % need the position of each ftr in the doi in the full configuration
            ftr_pos = find(obj.doi_ftrSequence(doi,:));    % this could be done ahead ot
            NF = sum(obj.doi_ftrSequence(doi,:)) ;
            NS = size(mc_labels,1);
            % initialize RR matrix and S vector
            RR = zeros(NF+1,NF+1);
            S = zeros(NF+1,NS);

            % make an array of ftr indices
            ftrs = 1:1:NF;

            % making a local phi_0
            ui_mcLabels = 0.5 * ones(NS,obj.L);
            logregs = obj.log_regs(doi,:);
            phi_0 = zeros(1,NS);
            for i = 1:NS
                expo = logregs(1) + dot(ui_mcLabels(i,:), logregs(2:end));
                p = 1.0/(1.0+exp( -1.0*expo));    
                phi_0(i) = 2*p-1.0;
            end
            % making a local phi_all
            % list of dois 
            all_mcLabels = mc_labels(:,obj.doi_mcs(doi,:));
            phi_all = zeros(1,NS);
            for i = 1:NS
                expo = logregs(1) + dot(all_mcLabels(i,:), logregs(2:end));
                p = 1.0/(1.0+exp( -1.0*expo));    
                phi_all(i) = 2*p-1.0;
            end 

            for s = 0:1:NF 
                if (s == 0)
                    tmpRR = zeros(NF+1,NF+1);
                    tmpRR(1,1) = 1;
                    RR = RR + tmpRR;
                    for i = 1:NS
                        S(:,i) = S(:,i) + phi_0(i) * ones(NF+1,1);
                    end
                elseif (s == NF)
                    tmpRR = ones(NF+1,NF+1);
                    RR = RR + tmpRR;
                    for i=1:NS
                        S(:,i) = S(:,i) + phi_all(i) * ones(NF+1,1);
                    end
                else 
                    factor = 1./(nchoosek(NF,s)*s*(NF-s));
                    sizeOfS = nchoosek(NF,s);
                    use = true(NF,1);
                    allSets = nchoosek(ftrs(use),s);
                    for i = 1:sizeOfS 
                        % get a configuration                     
                        currSequence = zeros(NF,1);
                        currSequence(allSets(i,:)) = 1;
                        % make z vector
                        z = [1 currSequence'];
                        % update matrix
                        RR = RR + (z'*z) * factor;
                        % get a classification
                        global_ftrs = ftr_pos(allSets(i,:));
                        u = 2*obj.doi_classification_from_subset(doi,global_ftrs,behavior,mc_labels) -1;
                        % update S vector
                        for jj = 1:NS
                            S(:,jj) = S(:,jj) + z' * factor * u(jj);
                        end
                    end
                end
            end

            SVs_local = obj.generate_LL_SVs_multiple_samples_spec0_all(RR,S,phi_all,NF,phi_0);

            SVs = zeros(obj.NF,NS);
            for i = 1:NF
                SVs(ftr_pos(i),:) = SVs_local(i,:);
            end

        end
        
        function [labels] = doi_classification_from_subset(obj, doi, subset, behavior, mc_labels)
            % gives a single drop out iteration's prediction given a feature subset and set of miniclassifier labels
            % input
            %           doi          drop out iteration
            %           subset       features subset
            %           behavior     behavior enum
            %           mc_labels    miniclassifier labels
            % output
            %           labels      drop out iteration regression output in (0,1) given inputs

            NS = size(mc_labels,1);
            mc_subset_labels = zeros(NS,obj.L);

            for i=1:obj.L
                mcIndex = obj.doi_mcs(doi,i);
                mcftrs = obj.mc_ftrs(mcIndex,1:obj.mc_ftr_number(mcIndex));
                tmp = sum ( ismember( mcftrs, subset ) );
                if (behavior == emptySetBehavior.mc_shallow )
                    if ( tmp == obj.mc_ftr_number(mcIndex) )
                        mc_subset_labels(:,i) = mc_labels(:,mcIndex);
                    else
                        mc_subset_labels(:,i) = 0.5 * ones(NS,1);
                    end
                elseif (behavior == emptySetBehavior.mc_deep )
                    if ( tmp == obj.mc_ftr_number(mcIndex) )
                        mc_subset_labels(:,i) = mc_labels(:,mcIndex);
                    elseif (tmp == 0)
                        mc_subset_labels(:,i) = 0.5 * ones(NS,1);
                    else
                        ftr = mcftrs( ismember(mcftrs,subset) );
                        mc_subset_labels(:,i) = mc_labels(:,ftr);
                    end
                else
                    disp( [ 'unimplemented behavior in doi_classification_from_subset' behaviour.Name]);
                end
            end

            labels = zeros(1,NS);
            logregs = obj.log_regs(doi,:);
            for j=1:NS
                expo = logregs(1) + dot( logregs(2:end) , mc_subset_labels(j,:) );
                labels(j) = 1.0/(1.0 + exp( -1.0 * expo));
            end

        end
        
        function [SVs,phi_0,phi_all,ftrs] = SVs_for_a_kNN(obj,imc,mc_labels)
            % SVs_for_a_kNN generates SVs for the features of a single kNN ie
            % mini-classifier
            %   uses written out formulations of the basic Shapley formual for NF = 1 or 2
            %   features
            % input
            %           imc     the current mini-classifier
            %           mc_labels    the mini-classifier labels for the sample
            % output
            %           SVs         (NF x NS) Shapley values
            %           phi_all     jsut the corresponding mc_label
            %           phi_0       always 0.5

            NF = obj.mc_ftr_number(imc);        % is 1 or 2
            NS = size(mc_labels,1);             % the number of samples
            SVs = zeros(NF,NS);

            if (NF == 1)                        % this mini-classifier is a single
                phi_0 = 0.5*ones(1,NS);
                phi_all = mc_labels(:,imc)';
                ftrs = obj.mc_ftrs(imc,1);
                SVs(1,:) = phi_all - phi_0;
            else                                % this mini-classifier is a pair
                phi_0 = 0.5*ones(1,NS);
                phi_all = mc_labels(:,imc)';
                ftrs = obj.mc_ftrs(imc,:);
                f1 = ftrs(1);
                f2 = ftrs(2);
                alpha = mc_labels(:,f1)' - mc_labels(:,f2)';
                SVs(1,:) = 0.5 * ( phi_all - phi_0 + alpha);
                SVs(2,:) = 0.5 * ( phi_all - phi_0 - alpha);
            end
        end
        
        function [SVs,phi_all,phi_0,alpha] = kNN_SVs_for_doi(obj,RR,doi,mc_labels)
            % kNN_SVs_for_doi SVS  calculates the SVs for the mini-classifiers of a doi
            % using LS algorithm (section 2.1)
            %   Input:
            %               RR:      precalculated RR matrices
            %               doi:     the doi
            %               mc_labels:  (NS x Nmcs) matrix of mini-classifire labels
            %               for a sample set
            %   Output: 
            %               SVs         Shapley values (NF x NS)
            %               phi_all     (1 x NS)
            %               phi_0       (1 x NS)

            NF = obj.L ;                    % the number of features is the leave-in number
            NS = size(mc_labels,1);         % number of samples
            mcs = obj.doi_mcs(doi,:);       % list of mini-classifiers used in this doi
            %  S vector
            S = zeros(NF+1,NS);             % SVs; note in reduced space of mini-classifiers

            % make an array of ftr indices
            ftrs = 1:1:NF;

            % making a local phi_0 and phi_all
            %%%%% making changes such that phi_0 for a doi is 0.5
            % the following could be pre-calculated
            logregs = obj.log_regs(doi,:);
            

            if ( sum(logregs(2:end)) == 0 )
                alpha = NaN;
                ui_labels = zeros(NS,obj.L);
            else
                alpha = -logregs(1)/sum(logregs(2:end));
                ui_labels = alpha * ones(NS,obj.L);
            end

            %%%%%
            % the phi_0 calculation is not necessary but kept in for a
            % check
            full_labels = mc_labels(:,mcs);
            
            phi_0 = zeros(1,NS);
            phi_all = zeros(1,NS);
            for i = 1:NS
                expo_0 = logregs(1) + dot(ui_labels(i,:), logregs(2:end));
                expo_1 = logregs(1) + dot(full_labels(i,:), logregs(2:end));
                p_0 = 1.0/(1.0+exp( -1.0*expo_0)); 
                p_1 = 1.0/(1.0+exp( -1.0*expo_1)); 
                phi_0(i) = 2*p_0-1.0;
                phi_all(i) = 2*p_1-1.0;
            end    

            for s = 0:1:NF    
                if (s == 0) 
                    for i = 1:NS
                        S(:,i) = S(:,i) + phi_0(i) * ones(NF+1,1);
                    end
                elseif (s == NF)
                    for i=1:NS
                        S(:,i) = S(:,i) + phi_all(i) * ones(NF+1,1);
                    end
                else            
                    factor = 1./(nchoosek(NF,s)*s*(NF-s));
                    sizeOfS = nchoosek(NF,s);
                    use = true(NF,1);
                    allSets = nchoosek(ftrs(use),s);
                    for i = 1:sizeOfS
                        currSequence = zeros(NF,1);
                        currSequence(allSets(i,:)) = 1;
                        % make z vector
                        z = [1 currSequence'];
                        % get a classification
                        currentSet = mcs(allSets(i,:));
                        red_labels = zeros(NS,obj.L);   % depending on the subset modify the mc_labels
                        for j = 1:obj.L
                            flag = sum( currentSet == mcs(j) );
                            if (flag > 0)
                                red_labels(:,j) = mc_labels(:,mcs(j));
                            else
                                red_labels(:,j) = ui_labels(:,1); 
                            end
                        end
                        u = zeros(1,NS);
                        for j=1:NS
                            expo = logregs(1) + dot( red_labels(j,:),logregs(2:end) );
                            u(j) = 2 * (1/(1+exp(-1*expo))) - 1;
                        end
                        % update S vector
                        for jj = 1:NS
                            S(:,jj) = S(:,jj) + z' * factor * u(jj);
                        end
                    end
                end
            end
            SVs = obj.generate_LL_SVs_multiple_samples_spec0_all(RR,S,phi_all,NF,phi_0);
        end
        
        function [RR] = pre_calculate_doi_RR(obj)
            %   pre_calculate_doi_RR generates RR matrix for a doi 
            %   these are the same for all dois as all dois use the same leave-in
            %   number L; this is described in section 2.1 formula 2.2

            NF = obj.L ;
            % initialize RR matrix and S vector
            RR = zeros(NF+1,NF+1);

            %make an array of ftr indices
            ftrs = 1:1:NF;

            for s = 0:1:NF 
                if (s == 0)
                    tmpRR = zeros(NF+1,NF+1);
                    tmpRR(1,1) = 1;
                    RR = RR + tmpRR;           
                elseif (s == NF)
                    tmpRR = ones(NF+1,NF+1);
                    RR = RR + tmpRR;  
                else
                    factor = 1./(nchoosek(NF,s)*s*(NF-s));
                    sizeOfS = nchoosek(NF,s);
                    use = true(NF,1);
                    allSets = nchoosek(ftrs(use),s);            
                    for i = 1:sizeOfS
                        % get a configuration                     
                        currSequence = zeros(NF,1);
                        currSequence(allSets(i,:)) = 1;
                        % make z vector
                        z = [1 currSequence'];
                        % update matrix
                        RR = RR + (z'*z) * factor;                
                    end
                end
            end
        end
    end
end

