
function [NO_SVs,phi_all] = NO_SVS(level_structure,mc_labels_dict,coeffs,ftrs_used_in_kNN, N_ftrs_used_in_kNN, verbose)
%NL_SVS: Calculates nested Owen SVs for a doi using
%   M. Besner, Discrete applied mathematics, 309 (2022)85--109 , using
%   algorithm 5.4 modified for dois and corrections to the paper
%           in line 12 S is a proper subset of B^k(i)
%           insert overbar(v)^0_i({i}) = SH_{i}(B^0|_B^1(i),tilde(v)^0_i)
%   Input:
%       level_structure:    A level structure (see class)
%       mc_labels_dict:     a containers.Map object giving the values of a
%                           kNN either -1 or +1; the keys are the arrays of
%                           the features used in a kNN converted to a
%                           string via num2str(ftrs_used_in_kNN(*,:)
%       coeffs:             the logreg expansion coefficients rescaled to
%                           work with values in(-1,+1) rather than (0,1)
%       ftrs_used_in_kNN:   the features used in the kNNs of the doi
%       N_ftrs_used_in_kNN: the number of ftrs used in a kNN
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
%   

    tmp     = mc_labels_dict.values;
    NS      = length(tmp{1});
    Nmcs    = size(ftrs_used_in_kNN,1);
    P = @(x) 2/(1+exp(-x)) - 1 ;        % predefine the softmax function

    % following NL SVs we calculate the SV of the middle layer components

    % how many components
    N_comp = length(level_structure.middlelayer);

    % form a list of all coalitions the middlelayer can form 
    % for later the first N_comp elements are the singletons
    % the value of this list is an index into the components of the middlelayer
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

    % make a value function for all of these; could merge with previous loop
    coalition_values = containers.Map;
    for i = 1:length(middlelayer_coalitions) 
        % grab the features over all components in the coalition
        coalition = middlelayer_coalitions{i};
        ftrs_in_coalition = cell2mat(level_structure.middlelayer{coalition(1)}.features);
        for j = 2:length(coalition)
            ftrs_in_coalition = [ftrs_in_coalition cell2mat(level_structure.middlelayer{coalition(j)}.features)];
        end
        [linear] = gen_logreg_argument(ftrs_in_coalition,ftrs_used_in_kNN,N_ftrs_used_in_kNN, NS,Nmcs,mc_labels_dict,coeffs);
        coalition_values(myfastint2str(sort(coalition))) = arrayfun(P,linear); %Note: the players are the middlelayer components
    end

    % calculate the SV for the components of the middlelayer
    % make use that the first N_comp elements of middlelayer.components are the
    % singletons and are numbered by their index in the middlelayer components
    % NOTE: it may be faster to use th e least square formulation here
    middlelayer_SVs = zeros(NS,N_comp);
    tau_i = middlelayer_coalitions(1:N_comp);
    for i = 1:N_comp
        t_i = tau_i{i};
        middlelayer_SVs(:,i) = SH(t_i,tau_i,coalition_values);
    end

    if (verbose )
        disp([' middlelayer SVs: ' num2str(middlelayer_SVs)])
    end
    
    % now we loop over every component to generate the SVs for features in this
    % component i.e. we go through the loops line 12 through 22 

    % grab the set of middle-layer feature sets
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
                % NEW
                S_ftrs{cnt} = singletons(tmp(j,:));
            end
        end
        
        %------------------- fix
        %S_ftrs = component_ftr_sets{ic};  % points to features of S via indices 1...N_elements
        %------------------- end fix

        % loop over these s_i in
        bar_v_11 = containers.Map;
        for i = 1:length(S)          % these will be singletons (in the bottomlayer)
            % replace the middlelayer component that contains ftr_i by s_i
            S_bki = component_ftr_sets;
            %S_bki{ic} = S{i};   bug, assumes S(i) are features
            % -------- fix
            S_bki{ic} = S_ftrs{i}; % they are singletons after all
            % -------- end fix
            % this gives a set of size length(middlelayer)
                % get all coalitions of this set and calculate their values ==>
                % tilde_v^1_1 s
                elementlist = 1:1:length(S_bki);
                tilde_v_11 = containers.Map;
                %for k = 1:N_elements   !!!! THIS IS just wrong
                for k = 1:length(S_bki)
                    tmp = nchoosek(elementlist,k);
                    for j = 1:size(tmp,1)
                        cnt = cnt + 1;
                        % calculate the value of the coalition and assign to
                        % dictionary
                        coalition = tmp(j,:);   % this points to numbers from 1:length(S_bki) ie indexes the elements of S_bki
                        ftrs_in_coalition = cell2mat(S_bki(tmp(j,:)));  % this has the features
                        [linear] = gen_logreg_argument(ftrs_in_coalition,ftrs_used_in_kNN,N_ftrs_used_in_kNN,NS,Nmcs,mc_labels_dict,coeffs);
                        tilde_v_11(myfastint2str(sort(coalition))) = arrayfun(P,linear); %Note: the players are the middlelayer components
                    end
                end
                % calculate the SV_s for S_i and assign to bar(v^_1)
                key = S{i};
                tau_i = num2cell(elementlist);
                try
                bar_v_11(myfastint2str(key)) = SH(tau_i{1},tau_i,tilde_v_11);
                catch
                    disp([])
                end
        end
        % assign the grand coalition for the features
        N_elements = length(component_ftr_sets{ic});
        bar_v_11(myfastint2str(1:1:N_elements)) = comp_SV;
        tau_i = num2cell(1:1:N_elements);
        %calculate the SH values for all features in the component
        ftrs = component_ftr_sets{ic};
        for i = 1:length(tau_i)
            t_i = tau_i{i};
            key = myfastint2str(ftrs(i));
            NO_SVs(key) = SH(t_i,tau_i,bar_v_11);
        end
    end

    % NOTE: populate phi_all
    gc = cell2mat(level_structure.bottomlayer);
    [linear] = gen_logreg_argument(gc,ftrs_used_in_kNN,N_ftrs_used_in_kNN,NS,Nmcs,mc_labels_dict,coeffs);
    phi_all = arrayfun(P,linear);
    
end

