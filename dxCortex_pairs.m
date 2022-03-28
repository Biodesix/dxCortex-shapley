classdef dxCortex_pairs < handle
    % class for a Diagnostic Cortex model using pairs of features in the mini-classifiers.
    % contains methods for training and predicting with the model and methods for 
    % calculating all 4 types of Shapely values from the paper.
    
    properties (GetAccess = public, SetAccess = private)
        NB                              % number of bags
        NF                              % number of features
        K                               % kNN K
        L                               % leave-in number
        Ndoi                            % number of drop-out iterations
        masters                         % (NB x 1) cell array of master_classifiers
        random_seed                     % overall random seed
        train_table                     % training ftr table (contains column 'Labels' )
    end
    
    methods (Access = public)
        
        function obj = dxCortex_pairs(NB,random_seed,Ndoi,L,K)
            obj.NB = NB;
            obj.random_seed = random_seed;
            obj.Ndoi = Ndoi;
            obj.L = L;
            obj.K = K;
        end
        
        function train(obj,ftr_table, inbag_fraction)
            
            obj.train_table = ftr_table;
            NS = size(ftr_table,1);
            obj.NF = size(ftr_table,2) -1;
            labels = ftr_table(:,'Labels');
            groups = unique(labels{:,:});
            G1_label = groups(1);
            G2_label = groups(2);
            samples = 1:1:NS ;
            
            G1_index = labels{:,:} == G1_label ;
            G2_index = labels{:,:} == G2_label ;
            G1_samples = samples( G1_index );
            G2_samples = samples( G2_index );
            NG1 = sum ( G1_index );
            NG2 = sum ( G2_index );
            NG1_train = floor(inbag_fraction*NG1) + 1;
            NG2_train = floor(inbag_fraction*NG2) + 1;
            NG1s = 1:1:NG1 ;
            NG2s = 1:1:NG2 ;
            
            obj.masters = cell(obj.NB,1);
            rng (obj.random_seed) ;
            for ib =1:obj.NB
                
                % make a master
                seed = randi(10000000);
                mc = masterclassifier(seed,obj.Ndoi,obj.L,obj.K);
                
                % train a master
                idx_g1 = NG1s( randperm(NG1,NG1_train) );
                idx_g2 = NG2s( randperm(NG2,NG2_train) );                
                t_g1 = G1_samples( idx_g1 );
                t_g2 = G2_samples( idx_g2 );                
                tmp = zeros(NS,1);
                tmp(t_g1) = 1;
                tmp(t_g2) = 1;
                
                train_data = ftr_table(logical(tmp),:);
                mc.train(train_data);
                
                obj.masters{ib} = mc;
                
            end
        end
        
        function labels = predict(obj,data)
            
            NS = size(data,1);
            labels = zeros(NS,1);
            for ib = 1:obj.NB
                tmp = obj.masters{ib}.predict(data);
                labels = labels + tmp;
            end
            
            labels = labels/obj.NB;
        end
        
        function [SVs,phi_all,phi_0] = SVs_shallow(obj,data)
            
            NS = size(data,1);
            SVs = zeros(obj.NF,NS);
            phi_all = zeros(1,NS);
            phi_0 = zeros(1,NS);
            for ib=1:obj.NB
                [tmp,p_all,p_0] = obj.masters{ib}.SVs_reverse_LS(data, emptySetBehavior.mc_shallow);
                phi_all = phi_all + p_all;
                phi_0 = phi_0 + p_0;
                SVs = SVs + tmp;
            end
            
            SVs = SVs/obj.NB;
            phi_0 = phi_0/obj.NB;
            phi_all = phi_all/obj.NB;
            
        end
        
        function [SVs,phi_all,phi_0] = SVs_deep(obj,data)
            
            NS = size(data,1);
            SVs = zeros(obj.NF,NS);
            phi_all = zeros(1,NS);
            phi_0 = zeros(1,NS);
            for ib=1:obj.NB
                [tmp,p_all,p_0] = obj.masters{ib}.SVs_reverse_LS(data, emptySetBehavior.mc_deep);
                phi_all = phi_all + p_all;
                phi_0 = phi_0 + p_0;
                SVs = SVs + tmp;
            end
            
            SVs = SVs/obj.NB;
            phi_0 = phi_0/obj.NB;
            phi_all = phi_all/obj.NB;
            
        end
        
        function [SVs,phi_all,phi_0] = SVs_restricted(obj,data)
            
            NS = size(data,1);
            SVs = zeros(obj.NF,NS);
            phi_all = zeros(1,NS);
            phi_0 = zeros(1,NS);
            for ib=1:obj.NB
                [tmp,p_all,p_0] = obj.masters{ib}.SVs_exact_LS(data, emptySetBehavior.doi);
                SVs = SVs + tmp;
                phi_all = phi_all + p_all';
                phi_0 = phi_0 + p_0';
            end
            
            SVs = SVs/obj.NB;
            phi_0 = phi_0/obj.NB;
            phi_all = phi_all/obj.NB;
            
        end
        
        function [SVs,phi_all,phi_0,alpha] = SVs_hierarchical(obj,data)
            
            NS = size(data,1);
            SVs = zeros(obj.NF,NS);
            phi_all = zeros(1,NS);
            phi_0 = zeros(1,NS);
            alpha = nan(obj.NB,1);
            for ib=1:obj.NB
                [tmp, p_all, p_0,alphaLogReg] = obj.masters{ib}.SVs_from_hierarchy(data);
                alpha(ib) = alphaLogReg;
                SVs = SVs + tmp;
                phi_all = phi_all + p_all;
                phi_0 = phi_0 + p_0;
            end
            
            SVs = SVs/obj.NB;
            phi_0 = phi_0/obj.NB;
            phi_all = phi_all/obj.NB;
            
        end
        
        function [SVs,phi_all,phi_0] = SVs_LS(obj,data)
 
            NS = size(data,1);
            SVs = zeros(obj.NF,NS);
            phi_all = zeros(1,NS);
            phi_0 = zeros(1,NS);
            for ib=1:obj.NB
                [tmp,p_all,p_0] = obj.masters{ib}.SVs_reverse_LS(data, emptySetBehavior.LS);
                SVs = SVs + tmp;
                phi_all = phi_all + p_all';
                phi_0 = phi_0 + p_0';
            end
            
            SVs = SVs/obj.NB;
            phi_0 = phi_0/obj.NB;
            phi_all = phi_all/obj.NB;
            
        end
        
        function [SVs,phi_all,phi_0] = SVs_NO(obj,data)
 
            NS = size(data,1);
            SVs = zeros(obj.NF,NS);
            phi_all = zeros(1,NS);
            phi_0 = zeros(1,NS);
            for ib=1:obj.NB
                [tmp,p_all,p_0] = obj.masters{ib}.SVs_reverse_LS(data, emptySetBehavior.NO);
                SVs = SVs + tmp;
                phi_all = phi_all + p_all';
                phi_0 = phi_0 + p_0';
            end
            
            SVs = SVs/obj.NB;
            phi_0 = phi_0/obj.NB;
            phi_all = phi_all/obj.NB;
            
        end
        
        function [SVs,phi_all,phi_0] = SVs_NL(obj,data)
 
            NS = size(data,1);
            SVs = zeros(obj.NF,NS);
            phi_all = zeros(1,NS);
            phi_0 = zeros(1,NS);
            for ib=1:obj.NB
                [tmp,p_all,p_0] = obj.masters{ib}.SVs_reverse_LS(data, emptySetBehavior.NL);
                SVs = SVs + tmp;
                phi_all = phi_all + p_all';
                phi_0 = phi_0 + p_0';
            end
            
            SVs = SVs/obj.NB;
            phi_0 = phi_0/obj.NB;
            phi_all = phi_all/obj.NB;
            
        end
        
    end
    
end

