function SH_val = SH(t_i,tau_i,v_tildes)
% SH calculates the SV for a player t_i in a player set tau_i using
% coalition function using the full Shapley formula and the formulation in 
% . Besner, Discrete applied mathematics, 309 (2022)85--109, Algorithm 5.1
% Input:
%       t_i:    an element of tau_i for which the SV is calculated
%       tau_i:  the set of players as a cell array of singletons
%       v_tildes:   the values of the coalition function
%               keys: string of players sorted by number
% Output:
%       SH_val: the SV for feature t_i
%NOTE: the empty coalition has value 0!!!!!!!!!!!!!!!!!!!
    
    tmp = v_tildes.values;
    NS = length(tmp{1});
    SH_val = zeros(NS,1);
    
    % make the set of players \ t_i
    S = cell(1,length(tau_i)-1);
    cnt = 0;
    for i = 1:length(tau_i)
        if ( ~ strcmp( myfastint2str(t_i) , myfastint2str(tau_i{i}) ) )
            cnt = cnt + 1;
            S{cnt} = tau_i{i};
        end
    end
    
    if (cnt ~= (length(tau_i) - 1) )
        error([' something is wrong in SH getting reducing features'])
    end
    
    N_S = length(S);
    elementlist = 1:1:N_S;
    
    for k = 1:N_S
        combinations = nchoosek(elementlist,k);
        for j = 1:size(combinations,1)
            coalition = [S{combinations(j,:)}];
            coalitionp = sort([t_i, coalition]);
            c_val = v_tildes(myfastint2str(coalition));
            c_val_p = v_tildes(myfastint2str(coalitionp));
            factor = factorial(k)*factorial(N_S-k)/ ... 
                factorial(N_S+1);
            SH_val = SH_val + factor*(c_val_p-c_val);
        end
    end
    
    % add the empty set term 
    factor = 1/(N_S+1);
    v_0 = 0;   %empty set coalition has value zero
    SH_val = SH_val + factor*(v_tildes(myfastint2str(t_i)) - v_0);
    
end

