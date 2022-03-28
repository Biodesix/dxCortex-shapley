function [SVs,phi_all] = LS_SVS(level_structure,mc_labels_dict,coeffs,ftrs_used_in_kNN, N_ftrs_used_in_kNN, verbose)
%NL_SVS: Calculates level structure SVs for a doi using
%   M. Besner, Discrete applied mathematics, 309 (2022)85--109 , using
%   algorithm 5.2 modified for dois 
%   Input:
%       level_structure:    A level structure (see class)
%       mc_labels_dict:     a containers.Map object giving the values of a
%                           kNN either -1 or +1; the keys are the arrays of
%                           the features used in a kNN converted to a
%                           string via num2str(ftrs_used_in_kNN(*,:)
%       coeffs:             the logreg expansion coefficients rescaled to
%                           work with values in(-1,+1) rather than (0,1)
%       ftrs_used_in_kNN:   the features used in the kNNs of the doi
%       verbose:            if 1 displays intermediate games
%   Output:
%       NO_SVs:             the nested level Shapley values as elements of
%                           a containers.Map with keys being the features
%                           as strings
%       phi_all:            the value of the grand coalition of all
%                           features in the doi
%   Note: This is using SH.m which calculates the SVs of game using the
%   direct formula. It might be faster to use the least squares formulation
%   when calculating the SVs for the components of some of the intermediate
%   games.

%   v1.0        HR                  3/13/2022
%   v1.1        HR                  3/16/2022 multiple fixes: 
%                                               -- deals with random ftr names
%   v1.2        HR                  3/19/2022 replace num2str and int2str
%   V1.3        HR                  3/20/2022  fix P; add # ftrs as arg to
%                                              gen_logreg_linear; add
%                                              N_ftrs_used_in_kNN as
%                                              argument
%   


    NF      = length(level_structure.bottomlayer);
    tmp     = mc_labels_dict.values;
    NS      = length(tmp{1});
    Nmcs    = size(ftrs_used_in_kNN,1);
    
    P = @(x) 2/(1+exp(-x)) - 1;        % predefine the softmax function
    
    SVs = containers.Map;
    
    for i = 1:NF
        
        % get the relevant coalitions and their coalition values
        % pick a feature from the bottomlayer
        ftr = level_structure.bottomlayer{i};

        % get the set of relevant coalitions for this feature
        R_1 = level_structure.LS_relevent_set(ftr);

        % get the values of the coalition functions for R_1
        coalition_values = containers.Map;
        for k = 1:length(R_1)
            ftrs_in_R_1 = R_1{k};
            linear = zeros(NS,1);
            linear(1:NS) = coeffs(1);
            for imc = 1:Nmcs
                ftrs_in_mc = ftrs_used_in_kNN(imc,1: N_ftrs_used_in_kNN(imc));
                overlap = intersect(ftrs_in_R_1,ftrs_in_mc);
                key = myfastint2str(sort(overlap));           % sort, maybe unnecessary
                if (mc_labels_dict.isKey(key))              % all other kNNs are 0
                    linear = linear + coeffs(imc+1)*mc_labels_dict(key);
                end
            end
            coalition_values(myfastint2str(sort(ftrs_in_R_1))) = arrayfun(P,linear); 
        end

        if (verbose )
            disp([ ' --------------------------- ' ])
            disp([ ' ftr =  ' num2str(ftr) ])
        end
        %-------------------------- k = 1 -----------------------------------------
        %calculate set T 
        if (verbose )
            disp([ ' k = 1  --------------------------- ' ])
        end
        
        % get the index of the ftr in the bottom layer
        ftr_index = cell2mat(level_structure.bottomlayer) == ftr; 

        % find the ancestors features and get its powerset
        B_1_of_ftr = cell2mat(level_structure.middlelayer{level_structure.ancestors(ftr_index)}.features);
        powset = powerset(B_1_of_ftr);

        % Intersection of Powerset with R_1
        % Matlab can't do an intersection of cell arrays of double arrays but can
        % deal with cell arrays of character vectors hence convert to cell arrays
        % of character vectors, do the intersection and convert back
        R_1_as_char = cellfun(@(x) myfastint2str(x),R_1,"UniformOutput",false);
        powerset_as_char = cellfun(@(x) myfastint2str(x),powset,"UniformOutput",false);
        tmp = intersect(powerset_as_char,R_1_as_char);
        T = cellfun(@(x) str2num(x),tmp,'UniformOutput',false);

        % now loop over elements of T to calculate vbar1 (T)
        vbar1s = containers.Map;
        for elements_T = 1:length(T)   
            t_i = T{elements_T};
        %     % make the set tau_i
        %         % take the children of the above layer B{3} i.e B{2}, these
        %         % are all elements of the k = 1 layer
        %         % find the child containing feature_1 
            tau_i = cell(1,length(level_structure.middlelayer));
            tau_i{level_structure.ancestors(ftr_index)} = t_i;  % replace by t_i
            for k = 1:length(level_structure.middlelayer)
                if ( k ~= level_structure.ancestors(ftr_index))
                    tau_i{k} = cell2mat(level_structure.middlelayer{k}.features);
                end
            end

            % now get all coalition values for players in tau_i
            %%% this is not working correctly for sets of sets need number of
            %%% components
            %%% rewrite by adding a layer of abstraction
            N_players = length(tau_i);
            % now get all coalition values for players in tau_i
            elementlist = 1:1:N_players;
            v_tildes = containers.Map;
            for k=1:length(tau_i)
                combinations = nchoosek(elementlist,k);
                for j = 1:size(combinations,1)
                    qs = [tau_i{combinations(j,:)}];
                    coalition_key = myfastint2str(sort(qs));          % points to features
                    key = myfastint2str(sort(combinations(j,:)));     % points to players
                    v_tildes(key) = coalition_values(coalition_key);
                end
            end
            % find the index of t_i in element_list
            index = -1;
            test = myfastint2str(sort(t_i));
            for ii = 1:N_players
                comp = myfastint2str(sort(tau_i{ii}));
                if (strcmp(test,comp))
                    index = ii;
                    break;
                end
            end
            if (index < 0)
                error([' could not find t_i in tau_i'])
            end
            vbar1s(myfastint2str(sort(t_i))) = SH(index,num2cell(elementlist),v_tildes);   %implementation of Shapley sum formula
            %print
            if (verbose )
                disp([ num2str(length(tau_i)) '-player-game: ' ])
                for k=1:length(tau_i)
                    disp(['   player ' num2str(k) ': ' num2str(tau_i{k})])
                end
%                 coalitions = v_tildes.keys;
%                 values = v_tildes.values;
%                 for k = 1:length(coalitions)
%                     disp(['   v(' coalitions{k} ') = ' num2str(values{k})])
%                 end
            end
            %disp(['   SH value for feature ' num2str(t_i) ' = ' num2str(vbar1s(num2str(t_i)) )])
        end
        
        %-------------------------- k = 0 -----------------------------------------
        T = ftr;
        if (verbose )
            disp([ ' k = 0  --------------------------- ' ])
        end
        % loop over elements of T to calculate vbar0 (T)
        vbar0s = containers.Map;
        for elements_T = 1:length(T)

            t_i = T;   % only one element cludge!!!!!
            % this one seems very simple
            tau_i = (level_structure.middlelayer{level_structure.ancestors(ftr_index)}.features);
            % now get all coalition values for players in tau_i
            % now get all coalition values for players in tau_i
            %%% this is not working correctly for sets of sets need number of
            %%% components
            %%% rewrite by adding a layer of abstraction
            N_players = length(tau_i);
            elementlist = 1:1:length(tau_i);
            v_tildes_0 = containers.Map;
            for k=1:length(tau_i)
                combinations = nchoosek(elementlist,k);
                for j = 1:size(combinations,1)
                    qs = [tau_i{combinations(j,:)}];
                    coalition_key = myfastint2str(sort(qs));              % points to features
                    key = myfastint2str(sort(combinations(j,:)));         % points to players
                    v_tildes_0(key) = vbar1s(coalition_key);
                end
            end
            % find the ftr = ftr in the elementlist; the elementlist numbers the
            % components of tau_i; 
            xxx = find(cell2mat(tau_i) == ftr);
            vbar0s(myfastint2str(sort(t_i))) = SH(xxx,num2cell(elementlist),v_tildes_0);   %implementation of Shapley sum formula
            if ( verbose )                
                disp([ num2str(length(tau_i)) '-player-game: ' ])
                for k=1:length(tau_i)
                    disp(['   player ' num2str(k) ': ' num2str(tau_i{k})])
                end
%                 coalitions = v_tildes_0.keys;
%                 values = v_tildes_0.values;
%                 for k = 1:length(coalitions)
%                     disp(['   v(' coalitions{k} ') = ' num2str(values{k})])
%                 end
%                 disp(['   SH value for feature ' num2str(t_i) ' = ' num2str(vbar0s(num2str(t_i)) )])
            end
        end
        key = myfastint2str(ftr);
        SVs(key) = vbar0s(myfastint2str(ftr));
    end

    key = myfastint2str(sort(cell2mat(level_structure.toplayer)));
    phi_all = coalition_values(key);
end

