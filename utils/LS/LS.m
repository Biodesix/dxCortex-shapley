classdef LS < handle
%LS  provides a level structure for a doi in the sense of 
%   M. Besner, Discrete applied mathematics, 309 (2022)85--109 
%   It is contructed from  the description of a doi takes care of disjointedness in constructor
%   uses k1nodes
%   v1.0        HR                  3/14/2022
    properties
        toplayer        % the grand coalition: cell array of vector of all features
        middlelayer     % mini-classifier layer: cell array of k1nodes
        bottomlayer     % all singletons: cell array of length number of features 
        ancestors       % a pointer to the ancestors of a singleton into middlelayer: double[]
    end
    
    methods
        function obj = LS(ftrs_used_in_kNN,N_ftrs_used_in_kNN)
            %Construct an instance of LS absed on doi descriptors
            %make the middle layer first, make sure each node has disjoint
            %feature sets
            obj.make_disjoint_nodes_from_mcs(ftrs_used_in_kNN,N_ftrs_used_in_kNN);
            %make a vector of all features
            all_ftrs = cell2mat(obj.middlelayer{1}.features);
            obj.ancestors = [repmat(1,1,length(all_ftrs))];
            for i = 2:length(obj.middlelayer) 
                all_ftrs = [ all_ftrs cell2mat(obj.middlelayer{i}.features) ];
                obj.ancestors = [obj.ancestors repmat(i,1,length(obj.middlelayer{i}.features))];
            end
            obj.toplayer = {all_ftrs};
            obj.bottomlayer = num2cell(all_ftrs);          
        end
        
        function make_disjoint_nodes_from_mcs(obj,ftrs_used_in_kNN,N_ftrs_used_in_kNN)
        % searches over all mini-classifiers and combinees those that share
        % features into k1nodes; all others remain nodes of their own

            N_mcs   = size(ftrs_used_in_kNN,1);     % number of mini-classifiers
            unused  = ones(1,N_mcs);                % initially all are unused
            list_of_k1nodes = cell(1,N_mcs);        % pre-assign a cell array
            cnt = 0;
            for i = 1:N_mcs     % loop over mini-classifier
                if (unused(i))
                    unused(i) = 0;
                    list = [i];
                    % loop over all , 1 or 2, ftrs in classifier i and
                    % search for another classifier that has these and add
                    % to a list of mini-classifiers
                    for j = 1:N_ftrs_used_in_kNN(i)
                        f_i = ftrs_used_in_kNN(i,j);
                        % loop over unused mcs
                        for k = (i+1):N_mcs
                            if (unused(k) && ismember(f_i,ftrs_used_in_kNN(k,:)) )
                                list = [list k];
                                unused(k) = 0;
                            end
                        end
                    end
                    % generate a node from these mini-classifiers 
                    tmp = k1node(list,ftrs_used_in_kNN, N_ftrs_used_in_kNN);
                    % add to a list of nodes
                    cnt = cnt + 1;
                    list_of_k1nodes{cnt} = tmp;
                end
            end
            list_of_k1nodes = list_of_k1nodes(1:cnt);
            % merge again 
            %   The above misses shared features, e.g 
            %       ftrs_used_in_kNN = [ 1 -1;1 2; 3 4; 3 5 ; 6 -1; 7 8; 5 9];
            %       N_ftrs_used_in_kNN = [1,2,2,2,1,2,2];
            % as when it merges 3 4 and 3 5 it sets unused of 3 5 to zero,
            % and hence it does not add 5 9 at this step
            % let's do another merge step 
            % NOTE: one could get rid of the unused(k) condition in line 50
            %       just adding another merge step on the node level does
            %       the trick though
            unused = ones(1,length(list_of_k1nodes));
            merge_list = cell(1,length(list_of_k1nodes));
            cnt = 0;
            for i = 1:length(list_of_k1nodes)
                if (unused(i))
                    unused(i) = 0;
                    tmp = list_of_k1nodes{i};
                    for k = i+1:length(list_of_k1nodes)
                        if (~isempty(intersect( cell2mat(list_of_k1nodes{i}.features),cell2mat(list_of_k1nodes{k}.features))))
                            tmp = tmp.add(list_of_k1nodes{k});
                            unused(k) = 0;
                        end
                    end
                    cnt = cnt + 1;
                    merge_list{cnt} = tmp;
                end
            end
            % assign to middlelayer property of class
            obj.middlelayer = merge_list(1:cnt);
        end
        
        function R_i = LS_relevent_set(obj,ftr)
            % makes the set of relevant coalitions for feature ftr for a level structure
            % using remark 5.2 in M. Besner, Discrete applied mathematics, 309 (2022)85--109
            
            % get the name of the ftr
            ftr_index = cell2mat(obj.bottomlayer) == ftr;            
            feature_1 = obj.bottomlayer{ftr_index};
            
            % find the feature and its siblings and initialize R_1
            R_1 = obj.middlelayer{obj.ancestors(ftr_index)}.features;
            
            % now on the middle level add all elements of the middlelayer that don't
            % contain feature_1
            
            size_R_1 = length(R_1);
            for i = 1:length(obj.middlelayer)
                if (~( i == obj.ancestors(ftr_index)) )
                    size_R_1 = size_R_1 +1;
                    R_1{size_R_1} = cell2mat(obj.middlelayer{i}.features);                    
                end
            end

            % now make all coalitions from the elements in R_1
            % use nchoosek of the numbers 1,2,..., size_R_1, ==> the possible sizes go
            % from 2 (we already have the singletons) to size_R_1
            elementlist = 1:size_R_1;    % positions in R_1
            R_2_size = size_R_1;
            for i = 2:size_R_1
                combinations = nchoosek(elementlist,i);
                for j=1:size(combinations,1)     % loop over all possible combinations
                    new_element = [];
                    for j1 = 1:size(combinations,2)
                        new_element = [ new_element  R_1{combinations(j,j1)}];   
                    end
                    R_2_size = R_2_size + 1;
                    R_1{R_2_size} = new_element;
                end
            end
 
            R_i = R_1;
            
        end
        
        function RR = set_of_middlelayer_sets(obj)
            % this function returns a set of middlelayer features, i.e. a
            % cell array of length = number of middlelayer nodes
            N_nodes = length(obj.middlelayer);
            RR = cell(1,N_nodes);
            for i = 1:N_nodes
                RR{i} = cell2mat(obj.middlelayer{i}.features);
            end           
        end
        
    end
end

