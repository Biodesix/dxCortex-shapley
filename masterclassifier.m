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

            if ( behavior == emptySetBehavior.doi || behavior == emptySetBehavior.mc_deep || behavior == emptySetBehavior.mc_shallow)
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
            else
                % make the label dictionary; 
                % Note: as per convention is
                % there is a request from a coalition for a singleton that
                % is not in the list of all mini-classifiers it is given
                % the value 0.
                mc_labels_dict = containers.Map;
                mc_labels_2 = zeros(size(samples,1),obj.Nmcs);
                % loop over all mini-classifiers
                for imc = 1:obj.Nmcs
                    ftrs_used = obj.mc_ftrs(imc,1:obj.mc_ftr_number(imc));
                    key = myfastint2str(sort(ftrs_used));
                    % !!!!!!!!!!!!!!!!!
                    mc_labels_dict(key) = 2*mc_labels(:,imc) - 1;
                    mc_labels_2(:,imc)  = 2*mc_labels(:,imc) - 1;
                end
                
                SVs = zeros(NF,NS);
                phi_all = zeros(1,NS);
                phi_0  = zeros(1,NS);
                % loop over dois
                for i = 1:obj.Ndoi
                    % get the SVs for a doi
                    [SVs_tmp, p_all, p_0] = level_structure_SVs_for_doi(obj,i,behavior,mc_labels_dict,mc_labels_2);
                    phi_all = phi_all + p_all;
                    phi_0 = phi_0 + p_0;
                    % update SVs
                    key_vals    = SVs_tmp.keys;
                    vals        = SVs_tmp.values;
                    for j = 1:length(SVs_tmp.keys)
                        key = str2num(key_vals{j});
                        SVs(key,:) = SVs(key,:) + (vals{j})';
                    end
                    %disp([])
                end
                SVs_all = SVs/obj.Ndoi;
                phi_0 = phi_0/obj.Ndoi;
                phi_all = phi_all/obj.Ndoi;
                
            end
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
        
        function [SVs_dict, phi_all, phi_0] = level_structure_SVs_for_doi(obj,doi,behavior,mc_labels_dict,mc_labels)
            
            % get parameters describing a doi from 
            % or from doi_mcs(Ndoi,L) and mc_ftrs(Nmcs,L)
            % !!!!! needs mc_labels with values in [-1,1]
            ftrs_used_in_kNN = zeros(obj.L,2);
            N_ftrs_used_in_kNN = zeros(1,obj.L);
            for i=1:obj.L
                mcIndex = obj.doi_mcs(doi,i);
                NF_in_mc = obj.mc_ftr_number(mcIndex);
                ftrs_used_in_kNN(i,:) = obj.mc_ftrs(mcIndex,1:NF_in_mc);
                N_ftrs_used_in_kNN(i) = NF_in_mc;
            end
            
            % rescale the coefficients
            % get the log-reg coefficients for this doi and rescale them (working with
            % -1 and +1 rather than 0 and 1
            coeffs          = obj.log_regs(doi,:);
            coeffs(1)       = coeffs(1) + 0.5*sum(coeffs(2:end));
            coeffs(2:end)   = 0.5 * coeffs(2:end);
            
            % make a level_structure
            level_structure = LS(ftrs_used_in_kNN,N_ftrs_used_in_kNN);
            
            verbose = 0;
            if (behavior == emptySetBehavior.NL)
                %[SVs_dict,phi_all] = NL_SVS(level_structure,mc_labels_dict,coeffs,ftrs_used_in_kNN, N_ftrs_used_in_kNN, verbose);
                [SVs_dict,phi_all,p0] = obj.NL_SVS_local(doi,level_structure,coeffs, mc_labels, verbose);
            elseif (behavior == emptySetBehavior.HNL)
                %[SVs_dict,phi_all,p0] = obj.HNL_SVS_local_2(doi,level_structure,coeffs, mc_labels, verbose);
                [SVs_dict,phi_all,p0] = obj.HNL_SVS_local(doi,level_structure,coeffs, mc_labels, verbose);
            elseif (behavior == emptySetBehavior.NO)
                %[SVs_dict,phi_all] = NO_SVS(level_structure,mc_labels_dict,coeffs,ftrs_used_in_kNN, N_ftrs_used_in_kNN, verbose);
                [SVs_dict,phi_all] = obj.NO_SVS_local(doi,level_structure,coeffs, mc_labels, verbose);
            elseif (behavior == emptySetBehavior.HNO)
                [SVs_dict,phi_all] = obj.HNO_SVS_local(doi,level_structure,coeffs, mc_labels, verbose);    
            elseif(behavior == emptySetBehavior.LS)
                disp([ 'WARNING: this is slow!!!!!!!'])
                [SVs_dict,phi_all] = LS_SVS(level_structure,mc_labels_dict,coeffs,ftrs_used_in_kNN, N_ftrs_used_in_kNN, verbose);               
            end
            
            % pass out phi_0
            tmp = exp(-coeffs(1));
            p0  = 2/(1+tmp) -1;
            tmp     = mc_labels_dict.values;
            NS      = length(tmp{1});
            phi_0 = p0*ones(NS,1);
            
        end
        
        function [NO_SVs,phi_all] = NO_SVS_local(obj,doi,level_structure, coeffs, mc_labels,verbose)
            %NO_SVS: Calculates nested Owen SVs for a doi using
%   M. Besner, Discrete applied mathematics, 309 (2022)85--109 , using
%   algorithm 5.4 modified for dois and corrections to the paper
%           in line 12 S is a proper subset of B^k(i)
%           insert overbar(v)^0_i({i}) = SH_{i}(B^0|_B^1(i),tilde(v)^0_i)
%   Input:
%       doi:                index into dois
%       level_structure:    A level structure (see class)
%       mc_labels:          array(#samples,#mcs) giving the values of a
%                           kNN either -1 or +1; 
%       coeffs:             the logreg expansion coefficients rescaled to
%                           work with values in(-1,+1) rather than (0,1)
%       verbose:            if 1 displays middlelayer SVs
%   Output:
%       NO_SVs:             the nested level Shapley values as elements of
%                           a containers.Map with keys being the features
%                           as strings
%       phi_all:            the value of the grand coalition of all
%                           features in the doi
%   Note: This is using SH.m which calculates the SVs of game using the
%   direct formula. It might be faster to use the least squares formulation
%   when calculating the SVs for the components of th e middlelayer

%   v1.0        HR                  3/13/2022
%   v1.1        HR                  3/16/2022 multiple fixes: 
%                                               -- deals with rnadom ftr names
%                                               -- error in line 132
%                                               N_elements -->
%                                               length(S_bki)
%                                               -- verbose == 1 displays
%                                               SVs for middlelayers
%   v1.2        HR                  3/19/2022 replace int2str and num2str
%   V1.3        HR                  3/20/2022  fix P; add # ftrs as arg to
%                                              gen_logreg_linear; add
%                                              N_ftrs_used_in_kNN as
%                                              argument
%   v1.4        HR                  3/28/2022 major fixes, use LS, etc
%   v1.5        HR                  4/2/2022 fixed normalization on middlelayer SVs    



            % following NL SVs we calculate the SV of the middle layer components

            % how many components
            N_comp = length(level_structure.middlelayer);
            NS      = size(mc_labels,1);
            P   = @(x) 2/(1+exp(-x)) - 1 ;
            v0  = P(coeffs(1));

            % form a list of all coalitions the middlelayer can form 
            elementlist = 1:1:N_comp;
            middlelayer_coalitions = cell(1,2^N_comp-1);
            cnt = 0;
            for k = 1:N_comp
                tmp = nchoosek(elementlist,k);
                for j = 1:size(tmp,1)
                    cnt = cnt + 1;
                    middlelayer_coalitions{cnt} = tmp(j,:);
                end
            end

            %----------- for SH_LS
            if (N_comp > 1 )
                ftrs_per_player = cell(N_comp,1);
                for i = 1:N_comp
                    ftrs_per_player{i} = cell2mat(level_structure.middlelayer{i}.features);
                end
                [middlelayer_SVs, phi_all, phi_0] = SH_LS(obj,doi,ftrs_per_player,mc_labels,coeffs,0);
            elseif (N_comp == 1)
                ftrs_in_coalition = cell2mat(level_structure.middlelayer{1}.features);
                middlelayer_SVs = doi_prediction_from_subset(obj,doi,ftrs_in_coalition,mc_labels,coeffs) - doi_prediction_from_subset(obj,doi,[],mc_labels,coeffs);
            else
                error([' N__SVs_local:: this should never happen N_comp = ' num2str(N_comp)])
            end
            
            if (verbose )
                disp([' middlelayer SVs: ' num2str(middlelayer_SVs)])
            end

            % now we loop over every component to generate the SVs for features in this
            % component i.e. we go through the loops line 12 through 22 
            component_ftr_sets = level_structure.set_of_middlelayer_sets();

            NO_SVs = containers.Map;
            for ic = 1:N_comp
                comp_SV = middlelayer_SVs(:,ic); % the value of a component from the previous calculation
                % calculate the coalitions S that the children of this component can form
                % apart from the grand coalition 
                N_elements = length(component_ftr_sets{ic});
                elementlist = 1:1:N_elements;
                S = cell(1,2^N_elements-2);  % do not include this one's grand coalition hence -2 instead of -1
                S_ftrs = cell(1,2^N_elements-2);
                singletons = component_ftr_sets{ic};
                cnt = 0;
                for k = 1:N_elements-1
                    tmp = nchoosek(elementlist,k);
                    for j = 1:size(tmp,1)
                        cnt = cnt + 1;
                        S{cnt} = tmp(j,:);  
                        S_ftrs{cnt} = singletons(tmp(j,:));
                    end
                end

                % loop over these s_i in
                bar_v_11 = containers.Map;
                for i = 1:length(S)          % these will be singletons (in the bottomlayer)
                    % replace the middlelayer component that contains ftr_i by s_i
                    S_bki = component_ftr_sets;
                    S_bki{ic} = S_ftrs{i}; % they are singletons after all
                    key = S{i};
                    %----------- for SH_LS
                    if (length(S_bki) > 1 )
                        ftrs_per_player = S_bki;
                        [ttt, phi_all, phi_0] = SH_LS(obj,doi,ftrs_per_player,mc_labels,coeffs,0);
                        bar_v_11(myfastint2str(key)) = ttt(:,ic);      % need the ic'th SV
                    elseif (length(S_bki) == 1)
                        ftrs_in_coalition = S_bki{1};
                        bar_v_11(myfastint2str(key)) = doi_prediction_from_subset(obj,doi,ftrs_in_coalition,mc_labels,coeffs) - doi_prediction_from_subset(obj,doi,[],mc_labels,coeffs);
                    else
                        error([' N__SVs_local:: this should never happen N_comp = ' num2str(N_comp)])
                    end
                end
                % assign the grand coalition for the features
                N_elements = length(component_ftr_sets{ic});
                bar_v_11(myfastint2str(1:1:N_elements)) = comp_SV + v0*ones(NS,1);
                tau_i = num2cell(1:1:N_elements);
                %calculate the SH values for all features in the component
                ftrs = component_ftr_sets{ic};
                [ttt_1, ~, ~] = SH_LS_from_dictionary(obj,doi,tau_i,mc_labels,coeffs,bar_v_11,0);

                for i = 1:length(tau_i)
                    key = myfastint2str(ftrs(i));
                    NO_SVs(key) = ttt_1(:,i);
                end
                
            end

            % NOTE: populate phi_all
            phi_all = doi_prediction_from_subset(obj,doi,cell2mat(level_structure.bottomlayer),mc_labels,coeffs);
    
        end
        
        function [linear] = gen_logreg_argument_local(obj,doi,NS,mc_labels,coeffs,subset)
        % gen_logreg_argument: calculate the linear part, argument of the logreg,
        %   for a coalition function
        % DEPRECADED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        % Input:
        %       ftrs_in_coalition:  Array of the features used in a coalition
        %       ftrs_used_in_kNN:   Array of the features used in a mini-classifier
        %       N_ftrs_used_in_kNN: Array containing the number of ftrs used in mc
        %       NS:                 Number of samples
        %       Nmcs:               Number of mini-classifiers in the doi
        %       mc_labels_dict:     dictionary of mini-classifier labels 
        %                           key: sorted string of features
        %                           value: array (over samples) of mini-classifier
        %                           labels
        %   v1.0        HR                  3/14/2022
        %   v1.1        HR                  3/19/2022
        %                                   replace int2str and num2st
        %   v1.2        HR                  3/19/2022
        %                                   hand code set intersection
        %   v1.3        HR                  3/20/2022
        %                                   include N_ftrs_used_in_kNN 
        %
            linear = zeros(NS,1);
%             linear(1:NS) = coeffs(1);
%             for imc = 1:Nmcs
%                 ftrs_in_mc = ftrs_used_in_kNN(imc,1:N_ftrs_used_in_kNN(imc));
%                 %overlap = intersect(ftrs_in_coalition,ftrs_in_mc);
%                 % handcode intersection for arrays of positive integers
%                 % NOTE !!!!! for large integers one may have to use another
%                 % indirection
%                 if ( ~isempty(ftrs_in_coalition) && ~isempty(ftrs_in_mc) )
%                     PP = zeros(1, max(max(ftrs_in_coalition),max(ftrs_in_mc)));
%                     PP(ftrs_in_coalition) = 1;
%                     overlap = ftrs_in_mc(logical(PP(ftrs_in_mc)));
%                 else
%                     overlap = [];
%                 end
%                 %key = num2str(sort(overlap));           % sort, maybe unnecessary
%                 if( ~isempty(overlap) )
%                     key = myfastint2str(sort(overlap)); 
%                     if (mc_labels_dict.isKey(key))              % all other kNNs are 0
%                         linear = linear + coeffs(imc+1)*mc_labels_dict(key);
%                     end
%                 end
%             end
            
            %translate the following
            NS = size(mc_labels,1);
            mc_subset_labels = zeros(NS,obj.L);

            for i=1:obj.L
                mcIndex = obj.doi_mcs(doi,i);
                mcftrs = obj.mc_ftrs(mcIndex,1:obj.mc_ftr_number(mcIndex));
                %NOTE: use setdiff instead of ismember to speed up
                tmp = sum ( ismember( mcftrs, subset ) );
                if ( tmp == obj.mc_ftr_number(mcIndex) )
                    mc_subset_labels(:,i) = mc_labels(:,mcIndex);
                elseif (tmp == 0)
                    mc_subset_labels(:,i) = zeros(NS,1);
                else
                    ftr = mcftrs( ismember(mcftrs,subset) );
                    mc_subset_labels(:,i) = mc_labels(:,ftr);
                end                
            end

            %labels = zeros(1,NS);
            %logregs = obj.log_regs(doi,:);
            for j=1:NS
                linear(j) = coeffs(1) + dot( coeffs(2:end) , mc_subset_labels(j,:) );
                %labels(j) = 1.0/(1.0 + exp( -1.0 * expo));
            end
        end
        
        function [values] = doi_prediction_from_subset(obj,doi,subset,mc_labels,coeffs)
            % given a list of ftrnames, subset,  generate the value of the logistic
            % regression using the deep convention; attempt at optimization
            % NOTE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            % Like the old code this relies on the fact that the NF singles
            % are in the first NF second indices of mc_labels, and are
            % numbered from 1:NF
        %   Input:
        %       doi:                used in ftrs_usedby_doi_and_mc
        %       subset:             the subset of ftrnames
        %       mc_labels:          array(#samples,#mcs) giving the values of a
        %                           kNN either -1 or +1; 
        %       coeffs:             the logreg expansion coefficients rescaled to
        %                           work with values in(-1,+1) rather than (0,1)
        %   Output:
        %       values:             logreg(subset)   (NS,1)
        %
        %   v1.0        HR                  3/28/2022
 
            % squeeze seems to be slow
            doi_ftrs_permc = zeros(obj.L,2);
            doi_ftrs_permc(:,:) = obj.ftrs_usedby_doi_and_mc(doi,:,:);
            check = false(size(doi_ftrs_permc,1),size(doi_ftrs_permc,2));
            for i = 1:length(subset)
                check = check | (doi_ftrs_permc == subset(i));
            end
            mc_type = zeros(obj.L,1);
            for i = 1:obj.L
                sw = check(i,1) + check(i,2);
                if (sw == 2)        % both ftrs are there
                    mc_type(i) = -1;
                elseif (sw == 1)    % only one feature is there
                    mc_type(i) = doi_ftrs_permc(i,check(i,:));
                end
            end
            
            linear = zeros(size(mc_labels,1),1);
            linear(:) = coeffs(1);
            for i = 1:obj.L
                if (mc_type(i) == -1)       % we have a pair
                    linear = linear + coeffs(i+1)*mc_labels(:,obj.doi_mcs(doi,i));
                elseif (mc_type(i) > 0)     % we have a single
                    linear = linear + coeffs(i+1)*mc_labels(:,mc_type(i));
                end
            end
            
            P = @(x) 2/(1+exp(-x)) - 1 ;        % predefine the softmax function
            values = arrayfun(P,linear);
        end
        
        function [SVs, phi_all, phi_0] = SH_LS(obj,doi,ftrs_per_player,mc_labels,coeffs,debug)
            % LS formalism SVs for a set of players with ftrs_per_player as
            % players; the values of the coalitions are calculated
        %   Input:
        %       doi:                not used; too lazy to change
        %       ftrs_per_player:    the players (array) in a cell array
        %       mc_labels:          array(#samples,#mcs) giving the values of a
        %                           kNN either -1 or +1; 
        %       coeffs:             the logreg expansion coefficients rescaled to
        %                           work with values in(-1,+1) rather than (0,1)
        %       debug:              if 1 enables a break point @ the end
        %   Output:
        %       SVs:                the Shapley values as array (NS,NF) the
        %                           features are counted as in ftrs_per_player
        %       phi_all:            the value of the grand coalition (NS,1)
        %       phi_0:              the value of the empty set (NS,1)
        %
        %   v1.0        HR                  3/28/2022
            % need the position of each ftr in the doi in the full configuration
            
            NF = length(ftrs_per_player);
            ftr_pos = 1:1:NF;    % this could be done ahead ot
            NS = size(mc_labels,1);
            % initialize RR matrix and S vector
            RR = zeros(NF+1,NF+1);
            S = zeros(NF+1,NS);

            % make an array of ftr indices
            ftrs = 1:1:NF;

            % making a local phi_0
            phi_0 = zeros(NS,1);
            phi_0(:) = 2/( 1+exp(-coeffs(1)) ) - 1 ;

            % making a local phi_all
            % list of dois
            gc = [ftrs_per_player{:}];
            phi_all = doi_prediction_from_subset(obj,doi,gc,mc_labels,coeffs);

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
                        ftrs_in_coalition = [ftrs_per_player{global_ftrs}];
                        u = doi_prediction_from_subset(obj,doi,ftrs_in_coalition,mc_labels,coeffs);
                        for jj = 1:NS
                            S(:,jj) = S(:,jj) + z' * factor * u(jj);
                        end
                    end
                end
            end

            SVs_local = obj.generate_LL_SVs_multiple_samples_spec0_all(RR,S,phi_all,NF,phi_0);
            
            SVs = SVs_local';
            
            if (debug )
                pp = 18;
            end

        end
        
        function [SVs, phi_all, phi_0] = SH_LS_from_dictionary(obj,doi,ftrs_per_player,mc_labels,coeffs,coalition_values,debug)
            % LS formalism SVs for a set of players with ftrs_per_player as
            % players; the values of the coalitions are stored in a map
        %   Input:
        %       doi:                not used; too lazy to change
        %       ftrs_per_player:    the players (array) in a cell array
        %       mc_labels:          array(#samples,#mcs) giving the values of a
        %                           kNN either -1 or +1; 
        %       coeffs:             the logreg expansion coefficients rescaled to
        %                           work with values in(-1,+1) rather than (0,1)
        %       coalition_values:   the values of all possible coalitions
        %                           of the players in a map, the keys for
        %                           these need to be present
        %       debug:              if 1 enables a break point @ the end
        %   Output:
        %       SVs:                the Shapley values as array (NS,NF) the
        %                           features are counted as in ftrs_per_player
        %       phi_all:            the value of the grand coalition (NS,1)
        %       phi_0:              the value of the empty set (NS,1)
        %
        %   v1.0        HR                  3/28/2022

            % need the position of each ftr in the doi in the full configuration         
            NF = length(ftrs_per_player);
            ftr_pos = 1:1:NF;    % somewhat silly !!!!!
            NS = size(mc_labels,1);
            % initialize RR matrix and S vector
            RR = zeros(NF+1,NF+1);
            S = zeros(NF+1,NS);

            % make an array of ftr indices
            ftrs = 1:1:NF;

            % making a local phi_0
            phi_0 = zeros(NS,1);
            phi_0(:) = 2/( 1+exp(-coeffs(1)) ) - 1 ;

            % making a local phi_all
            gc = [ftrs_per_player{:}];
            phi_all = coalition_values(myfastint2str(sort(gc)));

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
                        ftrs_in_coalition = [ftrs_per_player{global_ftrs}];
                        u = coalition_values(myfastint2str(sort(ftrs_in_coalition)));
                        for jj = 1:NS
                            S(:,jj) = S(:,jj) + z' * factor * u(jj);
                        end
                    end
                end
            end

            SVs_local = obj.generate_LL_SVs_multiple_samples_spec0_all(RR,S,phi_all,NF,phi_0);
            
            SVs = SVs_local';   % transpose from (NF,NS) to (NS,NF)
            
            if (debug )
                pp = 18;
            end

        end
        
        function [NL_SVs,phi_all,phi_0] = NL_SVS_local(obj,doi,level_structure,coeffs, mc_labels, verbose)
            %NL_SVS: Calculates nested level SVs for a doi using
        %   M. Besner, Discrete applied mathematics, 309 (2022)85--109 , using
        %   algorithm 5.3 modified for dois and simplified to get all ftrs' SV
        %   Input:
        %       level_structure:    A level structure (see class)
        %       mc_labels:          array(#samples,#mcs) giving the values of a
        %                           kNN either -1 or +1; 
        %       coeffs:             the logreg expansion coefficients rescaled to
        %                           work with values in(-1,+1) rather than (0,1)
        %       verbose:            if 1 displays the SVs of the middle layer
        %   Output:
        %       NL_SVs:             the nested level Shapley values as elements of
        %                           a containers.Map with keys being the features
        %                           as strings
        %       phi_all:            the value of the grand coalition of all
        %                           features in the doi
        %       phi_0:              the value of the empty set (NS,1)
        %
        %   v1.0        HR                  3/13/2022
        %   v1.1        HR                  3/16/2022
        %                                           -- added verbose == 1 option
        %   v1.2        HR                  3/19/2022  replaced num2str and int2str
        %   V1.3        HR                  3/20/2022  fix P; add # ftrs as arg to
        %                                              gen_logreg_linear; add
        %                                              N_ftrs_used_in_kNN as
        %                                              argument
        %   V1.4        HR                  3/28/2022  Multiple changes:
        %                                   --- use
        %                                   doi_prediction_from_subset for
        %                                   coalition value calculations
        %                                   --- use least squares for all
        %                                   internal SV calculations
        %   v1.5        HR                  4/2/2022 fixed normalization on middlelayer SVs    
        %

            NL_SVs  = containers.Map;
            NS      = size(mc_labels,1);
            
            P   = @(x) 2/(1+exp(-x)) - 1 ;
            v0  = P(coeffs(1));

            N_comp = length(level_structure.middlelayer);   % how many components

            % form a list of all coalitions the middlelayer can form 
            elementlist = 1:1:N_comp;
            middlelayer_coalitions = cell(1,2^N_comp-1);
            cnt = 0;
            for k = 1:N_comp
                tmp = nchoosek(elementlist,k);
                for j = 1:size(tmp,1)
                    cnt = cnt + 1;
                    middlelayer_coalitions{cnt} = tmp(j,:);
                end
            end

            % middlelayer SVs
            if (N_comp > 1 )
                ftrs_per_player = cell(N_comp,1);
                for i = 1:N_comp
                    ftrs_per_player{i} = cell2mat(level_structure.middlelayer{i}.features);
                end
                [middlelayer_SVs, ~, ~] = SH_LS(obj,doi,ftrs_per_player,mc_labels,coeffs,0);
            elseif (N_comp == 1)
                ftrs_in_coalition = cell2mat(level_structure.middlelayer{1}.features);
                middlelayer_SVs = doi_prediction_from_subset(obj,doi,ftrs_in_coalition,mc_labels,coeffs) - doi_prediction_from_subset(obj,doi,[],mc_labels,coeffs);
            else
                error([' N__SVs_local:: this should never happen N_comp = ' num2str(N_comp)])
            end
            if (verbose )
                disp([' middlelayer SVs: ' num2str(middlelayer_SVs)])
            end

            % bottomlayer for final SVs          
            for ic = 1:N_comp % loop over components            
                ftrs_in_component = level_structure.middlelayer{ic}.features;       % get the features in this component
                % make a dictionary from numbers to features so we can map back to
                % original feature names
                number2features = containers.Map;
                for i = 1:length(ftrs_in_component)
                    number2features(myfastint2str(i)) = ftrs_in_component{i};      
                end        
                c_vals = containers.Map;    %make the coalition values             
                c_vals(myfastint2str(1:1:length(ftrs_in_component))) = middlelayer_SVs(:,ic) + v0*ones(NS,1);     % take the grand coalition from above
                elementlist = 1:1:length(ftrs_in_component);   
                for k = 1:length(ftrs_in_component) - 1     % not the grand coalition
                    tmp = nchoosek(elementlist,k);
                    for j = 1:size(tmp,1)
                        coalition = sort(tmp(j,:));   %values from 1:length(ftrs_in_component)
                        ftrs_used_in_coalition = zeros(1,length(coalition));
                        for i = 1:length(coalition)
                            ftrs_used_in_coalition(i) = number2features(myfastint2str(coalition(i)));   % looks like array access but in fact it is dictionary access
                        end
                        c_vals(myfastint2str((coalition))) = doi_prediction_from_subset(obj,doi,ftrs_used_in_coalition,mc_labels,coeffs);
                    end
                end

                % calculate the SVs (map back to original feature names )
                tau_i = num2cell(1:1:length(ftrs_in_component));
                [ttt_1, ~, ~] = SH_LS_from_dictionary(obj,doi,tau_i,mc_labels,coeffs,c_vals,0);
                for i = 1:length(tau_i)
                    NL_SVs(myfastint2str(number2features(myfastint2str(tau_i{i})))) = ttt_1(:,i);
                end
             
            end

            %phi_0,phi_all
            phi_0 = v0*ones(NS,1); 
            phi_all = doi_prediction_from_subset(obj,doi,cell2mat(level_structure.bottomlayer),mc_labels,coeffs);

        end
        
        function [NL_SVs,phi_all,phi_0] = HNL_SVS_local(obj,doi,level_structure,coeffs, mc_labels, verbose)
            %NLH_SVS: Calculates nested level SVs for a doi using
        %   M. Besner, Discrete applied mathematics, 309 (2022)85--109 , using
        %   algorithm 5.3 modified for dois and simplified to get all ftrs' SV
        %   using the linear combination of components on the h- 0 level
        %   Input:
        %       level_structure:    A level structure (see class)
        %       mc_labels:          array(#samples,#mcs) giving the values of a
        %                           kNN either -1 or +1; 
        %       coeffs:             the logreg expansion coefficients rescaled to
        %                           work with values in(-1,+1) rather than (0,1)
        %       verbose:            if 1 displays the SVs of the middle layer
        %   Output:
        %       NL_SVs:             the nested level Shapley values as elements of
        %                           a containers.Map with keys being the features
        %                           as strings
        %       phi_all:            the value of the grand coalition of all
        %                           features in the doi
        %       phi_0:              the value of the empty set (NS,1)
        %
        %   v1.0        HR                  3/29/2022
        %

            NL_SVs  = containers.Map;
            NS      = size(mc_labels,1);
            
            P   = @(x) 2/(1+exp(-x)) - 1 ;
            v0  = P(coeffs(1));

            N_comp = length(level_structure.middlelayer);   % how many components

            % form a list of all coalitions the middlelayer can form 
            elementlist = 1:1:N_comp;
            middlelayer_coalitions = cell(1,2^N_comp-1);
            cnt = 0;
            for k = 1:N_comp
                tmp = nchoosek(elementlist,k);
                for j = 1:size(tmp,1)
                    cnt = cnt + 1;
                    middlelayer_coalitions{cnt} = tmp(j,:);
                end
            end

            % middlelayer SVs
            if (N_comp > 1 )
                ftrs_per_player = cell(N_comp,1);
                for i = 1:N_comp
                    ftrs_per_player{i} = cell2mat(level_structure.middlelayer{i}.features);
                end
                [middlelayer_SVs, ~, ~] = SH_LS(obj,doi,ftrs_per_player,mc_labels,coeffs,0);
            elseif (N_comp == 1)
                ftrs_in_coalition = cell2mat(level_structure.middlelayer{1}.features);
                middlelayer_SVs = doi_prediction_from_subset(obj,doi,ftrs_in_coalition,mc_labels,coeffs) - doi_prediction_from_subset(obj,doi,[],mc_labels,coeffs);
            else
                error([' N__SVs_local:: this should never happen N_comp = ' num2str(N_comp)])
            end
            if (verbose )
                disp([' middlelayer SVs: ' num2str(middlelayer_SVs)])
            end

            % bottomlayer for final SVs          
            for ic = 1:N_comp % loop over components            
                ftrs_in_component = level_structure.middlelayer{ic}.features;       % get the features in this component
                % make a dictionary from numbers to features so we can map back to
                % original feature names
                number2features = containers.Map;
                for i = 1:length(ftrs_in_component)
                    number2features(myfastint2str(i)) = ftrs_in_component{i};      
                end        
                c_vals = containers.Map;    %make the coalition values             
                %c_vals(myfastint2str(1:1:length(ftrs_in_component))) = middlelayer_SVs(:,ic);     % take the grand coalition from above
                c_vals(myfastint2str(1:1:length(ftrs_in_component))) = doi_prediction_from_subset_linear(obj,doi,cell2mat(ftrs_in_component),mc_labels,coeffs);
                elementlist = 1:1:length(ftrs_in_component);   
                for k = 1:length(ftrs_in_component) - 1     % not the grand coalition
                    tmp = nchoosek(elementlist,k);
                    for j = 1:size(tmp,1)
                        coalition = sort(tmp(j,:));   %values from 1:length(ftrs_in_component)
                        ftrs_used_in_coalition = zeros(1,length(coalition));
                        for i = 1:length(coalition)
                            ftrs_used_in_coalition(i) = number2features(myfastint2str(coalition(i)));   % looks like array access but in fact it is dictionary access
                        end
                        c_vals(myfastint2str((coalition))) = doi_prediction_from_subset_linear(obj,doi,ftrs_used_in_coalition,mc_labels,coeffs);
                    end
                end

                % calculate the SVs (map back to original feature names )
                tau_i = num2cell(1:1:length(ftrs_in_component));
                [ttt_1, phi_all, phi_0] = SH_LS_from_dictionary_linear(obj,doi,tau_i,mc_labels,coeffs,c_vals,0);
                
                tmp = middlelayer_SVs(:,ic) ./ (phi_all(:) - phi_0(:));
                for i = 1:length(tau_i)
                    NL_SVs(myfastint2str(number2features(myfastint2str(tau_i{i})))) = ttt_1(:,i) .* tmp(:);
                end
             
            end
            
            

            %phi_0,phi_all
            phi_0 = v0*ones(NS,1); 
            phi_all = doi_prediction_from_subset(obj,doi,cell2mat(level_structure.bottomlayer),mc_labels,coeffs);

        end
        
        function [values] = doi_prediction_from_subset_linear(obj,doi,subset,mc_labels,coeffs)
            % given a list of ftrnames, subset,  generate the value of the logistic
            % regression using the deep convention; attempt at optimization
            % NOTE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            % Like the old code this relies on the fact that the NF singles
            % are in the first NF second indices of mc_labels, and are
            % numbered from 1:NF
        %   Input:
        %       doi:                not used; too lazy to change
        %       subset:             the subset of ftrnames
        %       mc_labels:          array(#samples,#mcs) giving the values of a
        %                           kNN either -1 or +1; 
        %       coeffs:             the logreg expansion coefficients rescaled to
        %                           work with values in(-1,+1) rather than (0,1)
        %   Output:
        %       values:             logreg(subset)   (NS,1)
        %
        %   v1.0        HR                  3/29/2022
 
            % squeeze seems to be slow
            doi_ftrs_permc = zeros(obj.L,2);
            doi_ftrs_permc(:,:) = obj.ftrs_usedby_doi_and_mc(doi,:,:);
            check = false(size(doi_ftrs_permc,1),size(doi_ftrs_permc,2));
            for i = 1:length(subset)
                check = check | (doi_ftrs_permc == subset(i));
            end
            mc_type = zeros(obj.L,1);
            for i = 1:obj.L
                sw = check(i,1) + check(i,2);
                if (sw == 2)        % both ftrs are there
                    mc_type(i) = -1;
                elseif (sw == 1)    % only one feature is there
                    mc_type(i) = doi_ftrs_permc(i,check(i,:));
                end
            end
            
            linear = zeros(size(mc_labels,1),1);
            linear(:) = coeffs(1);
            for i = 1:obj.L
                if (mc_type(i) == -1)       % we have a pair
                    linear = linear + coeffs(i+1)*mc_labels(:,obj.doi_mcs(doi,i));
                elseif (mc_type(i) > 0)     % we have a single
                    linear = linear + coeffs(i+1)*mc_labels(:,mc_type(i));
                end
            end
            
            values = linear;
        end
        
        function [SVs, phi_all, phi_0] = SH_LS_from_dictionary_linear(obj,doi,ftrs_per_player,mc_labels,coeffs,coalition_values,debug)
            % LS formalism SVs for a set of players with ftrs_per_player as
            % players; the values of the coalitions are stored in a map
            % phi is 0 is from linear, NOTE make sure you call it with
            % linear coalition values
        %   Input:
        %       doi:                not used; too lazy to change
        %       ftrs_per_player:    the players (array) in a cell array
        %       mc_labels:          array(#samples,#mcs) giving the values of a
        %                           kNN either -1 or +1; 
        %       coeffs:             the logreg expansion coefficients rescaled to
        %                           work with values in(-1,+1) rather than (0,1)
        %       coalition_values:   the values of all possible coalitions
        %                           of the players in a map, the keys for
        %                           these need to be present
        %       debug:              if 1 enables a break point @ the end
        %   Output:
        %       SVs:                the Shapley values as array (NS,NF) the
        %                           features are counted as in ftrs_per_player
        %       phi_all:            the value of the grand coalition (NS,1)
        %       phi_0:              the value of the empty set (NS,1)
        %
        %   v1.0        HR                  3/29/2022

            % need the position of each ftr in the doi in the full configuration         
            NF = length(ftrs_per_player);
            ftr_pos = 1:1:NF;    % somewhat silly !!!!!
            NS = size(mc_labels,1);
            % initialize RR matrix and S vector
            RR = zeros(NF+1,NF+1);
            S = zeros(NF+1,NS);

            % make an array of ftr indices
            ftrs = 1:1:NF;

            % making a local phi_0
            phi_0 = zeros(NS,1);
            phi_0(:) = coeffs(1) ;

            % making a local phi_all
            gc = [ftrs_per_player{:}];
            phi_all = coalition_values(myfastint2str(sort(gc)));

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
                        ftrs_in_coalition = [ftrs_per_player{global_ftrs}];
                        u = coalition_values(myfastint2str(sort(ftrs_in_coalition)));
                        for jj = 1:NS
                            S(:,jj) = S(:,jj) + z' * factor * u(jj);
                        end
                    end
                end
            end

            SVs_local = obj.generate_LL_SVs_multiple_samples_spec0_all(RR,S,phi_all,NF,phi_0);
            
            SVs = SVs_local';   % transpose from (NF,NS) to (NS,NF)
            
            if (debug )
                pp = 18;
            end

        end
        
        function [NO_SVs,phi_all] = HNO_SVS_local(obj,doi,level_structure, coeffs, mc_labels,verbose)
            %HNO_SVS: Calculates nested Owen SVs for a doi using
%   M. Besner, Discrete applied mathematics, 309 (2022)85--109 , using
%   algorithm 5.4 modified for dois and corrections to the paper; this uses
%   the linear part of the logreg for middlelayer to lower layer
%   calculations
%           in line 12 S is a proper subset of B^k(i)
%           insert overbar(v)^0_i({i}) = SH_{i}(B^0|_B^1(i),tilde(v)^0_i)
%   Input:
%       doi:                index into dois
%       level_structure:    A level structure (see class)
%       mc_labels:          array(#samples,#mcs) giving the values of a
%                           kNN either -1 or +1; 
%       coeffs:             the logreg expansion coefficients rescaled to
%                           work with values in(-1,+1) rather than (0,1)
%       verbose:            if 1 displays middlelayer SVs
%   Output:
%       NO_SVs:             the nested level Shapley values as elements of
%                           a containers.Map with keys being the features
%                           as strings
%       phi_all:            the value of the grand coalition of all
%                           features in the doi
%   Note: This is using SH.m which calculates the SVs of game using the
%   direct formula. It might be faster to use the least squares formulation
%   when calculating the SVs for the components of th e middlelayer

%   v1.0        HR                  3/30/2022


            % following NL SVs we calculate the SV of the middle layer components

            % how many components
            N_comp = length(level_structure.middlelayer);

            % form a list of all coalitions the middlelayer can form 
            elementlist = 1:1:N_comp;
            middlelayer_coalitions = cell(1,2^N_comp-1);
            cnt = 0;
            for k = 1:N_comp
                tmp = nchoosek(elementlist,k);
                for j = 1:size(tmp,1)
                    cnt = cnt + 1;
                    middlelayer_coalitions{cnt} = tmp(j,:);
                end
            end

            %----------- for SH_LS
            if (N_comp > 1 )
                ftrs_per_player = cell(N_comp,1);
                for i = 1:N_comp
                    ftrs_per_player{i} = cell2mat(level_structure.middlelayer{i}.features);
                end
                [middlelayer_SVs, phi_all, phi_0] = SH_LS(obj,doi,ftrs_per_player,mc_labels,coeffs,0);
            elseif (N_comp == 1)
                ftrs_in_coalition = cell2mat(level_structure.middlelayer{1}.features);
                middlelayer_SVs = doi_prediction_from_subset(obj,doi,ftrs_in_coalition,mc_labels,coeffs) - doi_prediction_from_subset(obj,doi,[],mc_labels,coeffs);
            else
                error([' N__SVs_local:: this should never happen N_comp = ' num2str(N_comp)])
            end
            
            if (verbose )
                disp([' middlelayer SVs: ' num2str(middlelayer_SVs)])
            end

            % now we loop over every component to generate the SVs for features in this
            % component i.e. we go through the loops line 12 through 22 
            component_ftr_sets = level_structure.set_of_middlelayer_sets();

            NO_SVs = containers.Map;
            for ic = 1:N_comp
                %comp_SV = middlelayer_SVs(:,ic); % the value of a component from the previous calculation
                % calculate the coalitions S that the children of this component can form
                % apart from the grand coalition 
                N_elements = length(component_ftr_sets{ic});
                elementlist = 1:1:N_elements;
                S = cell(1,2^N_elements-2);  % do not include this one's grand coalition hence -2 instead of -1
                S_ftrs = cell(1,2^N_elements-2);
                singletons = component_ftr_sets{ic};
                cnt = 0;
                for k = 1:N_elements-1
                    tmp = nchoosek(elementlist,k);
                    for j = 1:size(tmp,1)
                        cnt = cnt + 1;
                        S{cnt} = tmp(j,:);  
                        S_ftrs{cnt} = singletons(tmp(j,:));
                    end
                end

                % loop over these s_i in
                bar_v_11 = containers.Map;
                for i = 1:length(S)          % these will be coalitions of singletons (in the bottomlayer)
                    % replace the middlelayer component that contains ftr_i by s_i
                    S_bki = component_ftr_sets;
                    S_bki{ic} = S_ftrs{i}; % they are singletons after all
                    key = S{i};
                    %----------- for SH_LS
                    if (length(S_bki) > 1 )
                        ftrs_per_player = S_bki;
                        [ttt, phi_all, phi_0] = SH_LS_linear(obj,doi,ftrs_per_player,mc_labels,coeffs,0);
                        bar_v_11(myfastint2str(key)) = ttt(:,ic);      % need the ic'th SV
                    elseif (length(S_bki) == 1)
                        ftrs_in_coalition = S_bki{1};
                        bar_v_11(myfastint2str(key)) = doi_prediction_from_subset_linear(obj,doi,ftrs_in_coalition,mc_labels,coeffs) - doi_prediction_from_subset_linear(obj,doi,[],mc_labels,coeffs);
                    else
                        error([' N__SVs_local:: this should never happen N_comp = ' num2str(N_comp)])
                    end
                end
                % assign the grand coalition for the features
                N_elements = length(component_ftr_sets{ic});
                %bar_v_11(myfastint2str(1:1:N_elements)) = comp_SV;  for H
                %we need the grand coalition here
                bar_v_11(myfastint2str(1:1:N_elements)) = doi_prediction_from_subset_linear(obj,doi,component_ftr_sets{ic},mc_labels,coeffs);
                tau_i = num2cell(1:1:N_elements);
                %calculate the SH values for all features in the component
                ftrs = component_ftr_sets{ic};
                [ttt_1, pall,p0] = SH_LS_from_dictionary_linear(obj,doi,tau_i,mc_labels,coeffs,bar_v_11,0);

                tmp = middlelayer_SVs(:,ic) ./ (pall(:) - p0(:)); % if denominator = 0 then NO_SVs = 0 !
                for i = 1:length(tau_i)
                    key = myfastint2str(ftrs(i));
                    NO_SVs(key) = ttt_1(:,i) .* tmp(:);
                end
                
            end

            % NOTE: populate phi_all; this seems to be correct
            phi_all = doi_prediction_from_subset(obj,doi,cell2mat(level_structure.bottomlayer),mc_labels,coeffs);
    
        end
        
        function [SVs, phi_all, phi_0] = SH_LS_linear(obj,doi,ftrs_per_player,mc_labels,coeffs,debug)
            % LS formalism SVs for a set of players with ftrs_per_player as
            % players; the values of the coalitions are calculated from the
            % linear part
        %   Input:
        %       doi:                not used; too lazy to change
        %       ftrs_per_player:    the players (array) in a cell array
        %       mc_labels:          array(#samples,#mcs) giving the values of a
        %                           kNN either -1 or +1; 
        %       coeffs:             the logreg expansion coefficients rescaled to
        %                           work with values in(-1,+1) rather than (0,1)
        %       debug:              if 1 enables a break point @ the end
        %   Output:
        %       SVs:                the Shapley values as array (NS,NF) the
        %                           features are counted as in ftrs_per_player
        %       phi_all:            the value of the grand coalition (NS,1)
        %       phi_0:              the value of the empty set (NS,1)
        %
        %   v1.0        HR                  3/30/2022
            % need the position of each ftr in the doi in the full configuration
            
            NF = length(ftrs_per_player);
            ftr_pos = 1:1:NF;    % this could be done ahead ot
            NS = size(mc_labels,1);
            % initialize RR matrix and S vector
            RR = zeros(NF+1,NF+1);
            S = zeros(NF+1,NS);

            % make an array of ftr indices
            ftrs = 1:1:NF;

            % making a local phi_0
            phi_0 = zeros(NS,1);
            phi_0(:) = coeffs(1) ;

            % making a local phi_all
            % list of dois
            gc = [ftrs_per_player{:}];
            phi_all = doi_prediction_from_subset_linear(obj,doi,gc,mc_labels,coeffs);

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
                        ftrs_in_coalition = [ftrs_per_player{global_ftrs}];
                        u = doi_prediction_from_subset_linear(obj,doi,ftrs_in_coalition,mc_labels,coeffs);
                        for jj = 1:NS
                            S(:,jj) = S(:,jj) + z' * factor * u(jj);
                        end
                    end
                end
            end

            SVs_local = obj.generate_LL_SVs_multiple_samples_spec0_all(RR,S,phi_all,NF,phi_0);
            
            SVs = SVs_local';
            
            if (debug )
                pp = 18;
            end

        end
        
% function [NL_SVs,phi_all,phi_0] = HNL_SVS_local_2(obj,doi,level_structure,coeffs, mc_labels, verbose)
%             %NLH_SVS: Calculates nested level SVs for a doi using
%         %   M. Besner, Discrete applied mathematics, 309 (2022)85--109 , using
%         %   algorithm 5.3 modified for dois and simplified to get all ftrs' SV
%         %   using the linear combination of components on the h- 0 level
%         %   Input:
%         %       level_structure:    A level structure (see class)
%         %       mc_labels:          array(#samples,#mcs) giving the values of a
%         %                           kNN either -1 or +1; 
%         %       coeffs:             the logreg expansion coefficients rescaled to
%         %                           work with values in(-1,+1) rather than (0,1)
%         %       verbose:            if 1 displays the SVs of the middle layer
%         %   Output:
%         %       NL_SVs:             the nested level Shapley values as elements of
%         %                           a containers.Map with keys being the features
%         %                           as strings
%         %       phi_all:            the value of the grand coalition of all
%         %                           features in the doi
%         %       phi_0:              the value of the empty set (NS,1)
%         %
%         %   v1.0        HR                  3/29/2022
%         %
% 
%             NL_SVs  = containers.Map;
%             NS      = size(mc_labels,1);
%             
%             P   = @(x) 2/(1+exp(-x)) - 1 ;
%             v0  = P(coeffs(1));
% 
%             N_comp = length(level_structure.middlelayer);   % how many components
% 
%             % form a list of all coalitions the middlelayer can form 
%             elementlist = 1:1:N_comp;
%             middlelayer_coalitions = cell(1,2^N_comp-1);
%             cnt = 0;
%             for k = 1:N_comp
%                 tmp = nchoosek(elementlist,k);
%                 for j = 1:size(tmp,1)
%                     cnt = cnt + 1;
%                     middlelayer_coalitions{cnt} = tmp(j,:);
%                 end
%             end
% 
%             % middlelayer SVs
%             if (N_comp > 1 )
%                 ftrs_per_player = cell(N_comp,1);
%                 for i = 1:N_comp
%                     ftrs_per_player{i} = cell2mat(level_structure.middlelayer{i}.features);
%                 end
%                 [middlelayer_SVs, ~, ~] = SH_LS(obj,doi,ftrs_per_player,mc_labels,coeffs,0);
%             elseif (N_comp == 1)
%                 ftrs_in_coalition = cell2mat(level_structure.middlelayer{1}.features);
%                 middlelayer_SVs = doi_prediction_from_subset(obj,doi,ftrs_in_coalition,mc_labels,coeffs) - doi_prediction_from_subset(obj,doi,[],mc_labels,coeffs);
%             else
%                 error([' N__SVs_local:: this should never happen N_comp = ' num2str(N_comp)])
%             end
%             if (verbose )
%                 disp([' middlelayer SVs: ' num2str(middlelayer_SVs)])
%             end
% 
%             % bottomlayer for final SVs          
%             for ic = 1:N_comp % loop over components            
%                 ftrs_in_component = level_structure.middlelayer{ic}.features;       % get the features in this component
%                 % make a dictionary from numbers to features so we can map back to
%                 % original feature names
%                 number2features = containers.Map;
%                 for i = 1:length(ftrs_in_component)
%                     number2features(myfastint2str(i)) = ftrs_in_component{i};      
%                 end        
%                 c_vals = containers.Map;    %make the coalition values             
%                 c_vals(myfastint2str(1:1:length(ftrs_in_component))) = middlelayer_SVs(:,ic) +v0*ones(NS,1);     % take the grand coalition from above
%                 %c_vals(myfastint2str(1:1:length(ftrs_in_component))) = doi_prediction_from_subset_linear(obj,doi,cell2mat(ftrs_in_component),mc_labels,coeffs);
%                 elementlist = 1:1:length(ftrs_in_component);   
%                 for k = 1:length(ftrs_in_component) - 1     % not the grand coalition
%                     tmp = nchoosek(elementlist,k);
%                     for j = 1:size(tmp,1)
%                         coalition = sort(tmp(j,:));   %values from 1:length(ftrs_in_component)
%                         ftrs_used_in_coalition = zeros(1,length(coalition));
%                         for i = 1:length(coalition)
%                             ftrs_used_in_coalition(i) = number2features(myfastint2str(coalition(i)));   % looks like array access but in fact it is dictionary access
%                         end
%                         c_vals(myfastint2str((coalition))) = doi_prediction_from_subset_linear(obj,doi,ftrs_used_in_coalition,mc_labels,coeffs);
%                     end
%                 end
% 
%                 % calculate the SVs (map back to original feature names )
% %                 tau_i = num2cell(1:1:length(ftrs_in_component));
% %                 [ttt_1, phi_all, phi_0] = SH_LS_from_dictionary_linear(obj,doi,tau_i,mc_labels,coeffs,c_vals,0);
%                 
% %                 tmp = middlelayer_SVs(:,ic) ./ (phi_all(:) - phi_0(:));
% %                 for i = 1:length(tau_i)
% %                     NL_SVs(myfastint2str(number2features(myfastint2str(tau_i{i})))) = ttt_1(:,i) .* tmp(:);
% %                 end
%              
%                 % calculate the SVs (map back to original feature names )
%                 tau_i = num2cell(1:1:length(ftrs_in_component));
%                 [ttt_1, ~, ~] = SH_LS_from_dictionary(obj,doi,tau_i,mc_labels,coeffs,c_vals,0);
%                 for i = 1:length(tau_i)
%                     NL_SVs(myfastint2str(number2features(myfastint2str(tau_i{i})))) = ttt_1(:,i);
%                 end
% 
%             end
%             
%             
% 
%             %phi_0,phi_all
%             phi_0 = v0*ones(NS,1); 
%             phi_all = doi_prediction_from_subset(obj,doi,cell2mat(level_structure.bottomlayer),mc_labels,coeffs);
% 
%         end

    end
end

