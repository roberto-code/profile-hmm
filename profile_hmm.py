from collections import Counter

# Utility functions

def read_viterbi_learning(file_name):
    lines = [line.strip() for line in open(file_name)]
    n_iter = int(lines[0])
    obs = lines[2]
    alphabet = lines[4].replace('\t',' ').split(' ')
    states = lines[6].replace('\t',' ').split(' ')
    
    head = list( filter(None,lines[8].split('\t')) )
    idx_st = {e:i for i,e in enumerate(head)}
    t_mat = [[0 for e in head] for e in head]
    for line in lines[9:9+len(head)]:
        elems = list( filter(None,line.split('\t')) )
        state = elems[0]
        for i,e in enumerate(elems[1:]):
            t_mat[ idx_st[state] ][ idx_st[head[i]] ] = float(e)
            
    n = 9 + len(head) + 1
    head_alph = list( filter(None,lines[n].split('\t')) )
    idx_alph = {e:i for i,e in enumerate(head_alph)}
    e_mat = [[0 for e in head_alph] for e in head]
    for line in lines[n+1:]:
        elems = list( filter(None,line.split('\t')) )
        state = elems[0]
        for i,e in enumerate(elems[1:]):
            e_mat[ idx_st[state] ][ idx_alph[head_alph[i]] ] = float(e)
    return (n_iter, obs, alphabet, states, t_mat, e_mat)

    
def formatted(f): 
    str_f = format(f, '.3f').rstrip('0').rstrip('.')
    if str_f == '1':
        str_f = '1.0' # Format required by the autograder...
    return str_f

# Given the transition matrix, return state labels
def state_labels(t_mat):
    n = len(t_mat)
    labels = ['S','I0']
    i = 1
    while i < n/3:
        c = str(i)
        labels += ['M'+c,'D'+c,'I'+c]
        i += 1
    labels.append('E')
    return labels

def print_t_mat(mat, labels):
    print('\t', end='')
    for e in labels:
        print(e, '\t', sep='', end='')
    print()
    for i,row in enumerate(mat):
        print(labels[i],'\t', sep='', end='')
        for e in row:
            print(formatted(e),'\t', sep='', end='')
        print()
                  
def print_e_mat(mat, alphabet, labels):
    print('\t',end='')
    for e in alphabet:
        print(e, '\t', sep='', end='')
    print()
    for i,row in enumerate(mat):
        print(labels[i],'\t', sep='', end='')
        for e in row:
            print(formatted(e),'\t', sep='', end='')
        print()

################################################################
                
def profile_hmm(theta, alphabet, mult_align):
    n = len(mult_align[0])
    m_cols = [False for e in mult_align] # meaningful columns
    for i,col in enumerate(mult_align):
        n_sp_symbols = sum(1 for e in col if e == '-')
        if n_sp_symbols / n < theta:
            m_cols[i] = True

    m = 3 * sum(m_cols) + 3 # M,D,I and S,E I0
    # Transition mat   
    t_mat = [[0 for i in range(m)] for j in range(m)]
    # Emission mat
    e_mat = [[0 for e in alphabet] for i in range(m)]
    idx_symb = {e:i for i,e in enumerate(alphabet)}
    
    for i in range(n):
        st = 0 # S
        for j in range(len(mult_align)):
            symbol = mult_align[j][i]
            next_st = st
            if m_cols[j]:
                if symbol != '-': # Transition to M
                    next_st +=1
                    while (next_st - 2) % 3 != 0:
                        next_st += 1
                    t_mat[st][next_st] += 1
                    st = next_st
                    e_mat[st][idx_symb[symbol]] += 1
                else: # Transition to D
                    # Add 2 to avoid transition from M_i to D_i
                    # instead of M_i to D_(i+1)
                    next_st += 2 
                    while next_st % 3 != 0:
                        next_st += 1
                    t_mat[st][next_st] += 1
                    st = next_st
            else:
                if symbol != '-':
                    if (st - 1) % 3 == 0: # Already in I state
                        t_mat[st][st] += 1
                    else:
                        next_st += 1
                        while (next_st - 1) % 3 != 0:
                            next_st += 1
                        t_mat[st][next_st] += 1
                        st = next_st
                    e_mat[st][idx_symb[symbol]] += 1
        t_mat[st][-1] += 1 # Transition to E

    # Normalize transition mat
    for i, e in enumerate(t_mat):
        s = sum(e)
        if s != 0:
            t_mat[i] = [x / s for x in e]
    # Normalize emission mat
    for i, e in enumerate(e_mat):
        s = sum(e)
        if s != 0:
            e_mat[i] = [x / s for x in e]

    return (t_mat, e_mat)

def add_pseudocount_t(t_mat, sigma):
    n = len(t_mat)
    t_mat[0][1] += sigma
    t_mat[0][2] += sigma
    t_mat[0][3] += sigma
    for i in range(1, n):
        if (i -1) % 3 == 0: # state I
            shift = 0
            while i+shift < n and shift < 3:
                t_mat[i][i+shift] += sigma  
                shift += 1
        elif (i - 2) % 3 == 0: # state M
            shift = 2
            while i+shift < n and shift < 5:
                t_mat[i][i+shift] += sigma  
                shift += 1
        elif i % 3 == 0: # state D
            shift = 1
            while i+shift < n and shift < 4:
                t_mat[i][i+shift] += sigma      
                shift += 1  
            
    # Normalize mat
    for i, e in enumerate(t_mat):
        s = sum(e)
        if s != 0:
            t_mat[i] = [x / s for x in e]       

def add_pseudocount_e(e_mat, sigma):
    n = len(e_mat)
    for i in range(1,n):
        # If state is not S,D or E
        if i != 0 and i %3 != 0 and i != n-1:
            e_mat[i] = [ e + sigma for e in e_mat[i]]

    # Normalize mat
    for i, e in enumerate(e_mat):
        s = sum(e)
        if s != 0:
            e_mat[i] = [x / s for x in e] 

def viterbi(obs, alphabet, n_states, trans_p, emit_p):
    V = [{}]
    path = {}
 
    # Initialize  first column
    V[0][0] = 1.
    path[0] = []
    states = [i for i in range(n_states-1)] # Dismiss E
    for st in states[1:]: # state 0 already filled
        if st % 3 == 0:
            V[0][st] = V[0][st-3] * trans_p[st-3][st]
            path[st] = path[st-3]+[st]
        else:
            V[0][st] = 0
            path[st] = [st]

    idx_symb = {e:i for i,e in enumerate(alphabet)}

    # Init second column
    V.append({})
    newpath = {}
 
    for y in states[1:]:
        (prob, state) = (0,0)
        if y % 3 == 0: # Deletion state
            (prob, state) = max((V[1][y0] * trans_p[y0][y], y0) for y0 in range(1,y)) # Up to current state
            # Path to a deletion state comes from a node in the same column
            newpath[y] = newpath[state] + [y]
        else:
            (prob, state) = max((V[0][y0] * trans_p[y0][y] * emit_p[y][ idx_symb[obs[0]] ], y0) for y0 in states)
            newpath[y] = path[state] + [y]
        V[1][y] = prob

    # Don't need to remember the old paths
    path = newpath
    
    # Run Viterbi for t > 1
    for t in range(2, len(obs)+1):
        V.append({})
        newpath = {}
 
        for y in states[1:]:
            (prob, state) = (0,0)
            if y % 3 == 0: # Deletion state
                (prob, state) = max((V[t][y0] * trans_p[y0][y], y0) for y0 in range(1,y)) # Up to current state
                newpath[y] = newpath[state] + [y]
            else:
                (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][ idx_symb[obs[t-1]] ], y0) for y0 in states[1:])
                newpath[y] = path[state] + [y]
            V[t][y] = prob
            
        # Don't need to remember the old paths
        path = newpath
        
        
    n = 0           # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t

    E = n_states - 1 # Transition to E node
    (prob, state) = max((V[n][y]* trans_p[y][E], y) for y in states[1:])
    return (prob, path[state])

def viterbi2(obs, states, trans_p, emit_p):
    V = [{}]
    path = {}
    
    start_p = 1 / len(states)
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p * emit_p[y][obs[0]]
        path[y] = [y]
 
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
 
        for y in states:
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
 

    n = 0           # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t

    (prob, state) = max((V[n][y], y) for y in states)
    return path[state]
 
def estimate_param(obs, alphabet, path, states):
    t_mat = [[0 for e in states] for e in states]
    for i in range(1, len(path)):
        t_mat[path[i-1]][path[i]] += 1
    
    # Normalize transition mat
    for i, e in enumerate(t_mat):
        s = sum(e)
        if s != 0:
            t_mat[i] = [x / s for x in e]
        else: # No transitions observed. all equiprobable
            l = len(e)
            t_mat[i] = [1 / l for x in e]

    e_mat = [[0 for y in alphabet] for e in states]  
    n_obs = len(obs)
    n_path = len(path)
    for i in range(n_path):
            e_mat[path[i]][obs[i]] += 1    

    # Normalize emission mat
    for i, e in enumerate(e_mat):
        s = sum(e)
        if s != 0:
            e_mat[i] = [x / s for x in e]
        else: # No emission observed. all equiprobable
            l = len(e)
            e_mat[i] = [1 / l for x in e]       
    
    return (t_mat, e_mat)  
    
def viterbi_learning(obs, alphabet, states, n_iter, t_mat, e_mat):
    for i in range(n_iter):
        #(p, path) = viterbi(obs, alphabet, len(states), t_mat, e_mat)
        path = viterbi2(obs, states, t_mat, e_mat)
        (t_mat, e_mat) = estimate_param(obs, alphabet, path, states)
    return (t_mat, e_mat)


def outcome_likelihood(start_p, obs, states, trans_p, emit_p):
    V = [{}]
 
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p * emit_p[y][obs[0]]
 
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
 
        for y in states:
            prob = sum(V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]] for y0 in states)
            V[t][y] = prob
 

    n = 0           # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t

    prob = sum(V[n][y] for y in states)
    return (prob, V)

def outcome_likelihood_back(obs, states, trans_p, emit_p):
    V = [{}]
 
    #When at the last index n the HMM must then transition to the sink, so p_t=1
    for y in states:
        V[0][y] = 1
 
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
 
        for y in states:
            prob = sum(V[t-1][y0] * trans_p[y][y0] * emit_p[y0][obs[t-1]] for y0 in states)
            V[t][y] = prob
 

    n = 0           # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t

    prob = sum(V[n][y] for y in states)
    return (prob, V)

#Baum-welch learning
def e_step(obs, states, t_mat, e_mat):
    m = len(obs)
    start_p = 1 / m
    (fwd_sink,V) = outcome_likelihood(start_p, obs, states, t_mat, e_mat)
    (x, V2) = outcome_likelihood_back(obs[::-1], states, t_mat, e_mat)

    PI = [ [ V[i][k] * V2[m-i-1][k] / fwd_sink for i in range(m)] for k in states]
    
    PI2 = [[[0 for i in range(m-1)] for k in states] for l in states]
    for l in states:
        for k in states:
            for i in range(m-1):
                forward_l_i = V[i][l]
                backward_k_i1 = V2[m-i-2][k]
                weight_l_k = t_mat[l][k] * e_mat[k][obs[i+1]]
                PI2[l][k][i] = forward_l_i * backward_k_i1 * weight_l_k / fwd_sink
    return (PI,PI2)
    
def m_step(obs, alphabet, states, PI, PI2):
    m = len(obs)
    t_mat = [[sum(PI2[l][k][i] for i in range(m-1)) for k in states] for l in states]
    e_mat = [[ sum(PI[k][i] if obs[i] == b else 0 for i in range(m)) for b in alphabet] for k in states]

    # Normalize t_mat
    for i, e in enumerate(t_mat):
        s = sum(e)
        if s != 0:
            t_mat[i] = [x / s for x in e]   
    # Normalize e_mat
    for i, e in enumerate(e_mat):
        s = sum(e)
        if s != 0:
            e_mat[i] = [x / s for x in e] 
            
    return (t_mat, e_mat)
    
    
            
if __name__ == "__main__":
    
    (n_iter, obs, alphabet, states, t_mat, e_mat) = read_viterbi_learning('data/test.txt') # same input format as baum-welch
    idx_alph = {e:i for i,e in enumerate(alphabet)}
    idx_obs = [ idx_alph[e] for e in obs]
    idx_states = [i for i in range(len(states))]
    idx_alph = [i for i in range(len(alphabet))]

    for i in range(n_iter):    
        (PI,PI2) = e_step(idx_obs, idx_states, t_mat, e_mat)
        (t_mat, e_mat) = m_step(idx_obs, idx_alph, idx_states, PI, PI2)
        
    print_t_mat(t_mat, states)
    print('--------')
    print_e_mat(e_mat, alphabet, states)
    
    
    
    


