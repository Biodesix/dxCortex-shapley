function [list_of_k1nodes] = make_disjoint_nodes_from_mcs(ftrs_used_in_kNN,N_ftrs_used_in_kNN)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    N_mcs = size(ftrs_used_in_kNN,1);
    unused = ones(1,N_mcs);
    list_of_k1nodes = cell(1,N_mcs);
    
    cnt = 0;
    for i = 1:N_mcs
        
        if (unused(i))
            
            unused(i) = 0;
            list = [i];
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
            tmp = k1node(list,ftrs_used_in_kNN, N_ftrs_used_in_kNN);
            cnt = cnt + 1;
            list_of_k1nodes{cnt} = tmp;
            
        end
        
    end
    
    list_of_k1nodes = list_of_k1nodes(1:cnt);
    
    % merge again 
    
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
    
    list_of_k1nodes = merge_list(1:cnt);
end

