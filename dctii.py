from scipy.linalg import circulant
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

class Graph(object):
    def __init__(self, n_vertices, edges):
        self.n_vertices = n_vertices
        self.vertices = np.arange(1,n_vertices+1)
        self.edges = edges # adjacency list implemented as dictionary mapping vertex (integer 1,...,n_vertices) to list of neighbors
        self.L = None
        self.A = None
        self.D = None
    def get_laplacian_adjacency_degree_matrix(self):
        if self.L is None:
            n = self.n_vertices
            D = np.zeros((n,))
            A = np.zeros((n, n))
            for i in np.arange(n):
                neighbors = self.edges.get(i+1,[])
                print("node", i+1, "neighbors", neighbors)
                for nbr, wgt in neighbors:
                    A[i,nbr-1] = wgt
                    D[i] += wgt
            self.A = A
            self.D = np.diag(D)
            self.L = self.D-self.A
        return self.L, self.A, self.D
    def __str__(self):
        return f"{self.n_vertices}__{self.edges}"
    def tps_smooth(self, data_locs, data_vals, m=2, lambda_=0.01):
        assert len(data_locs) == self.n_vertices
        if L is not None:
            L, A, D = self.get_laplacian_adjacency_degree_matrix()
            Lpinvm = np.linalg.pinv(L)**m
            K_g = Lpinvm[data_locs,:][:,data_locs]
            K = np.block([[K_g + lambda_*np.eye(len(samples)), np.ones((len(samples),1))], [np.ones((1,len(samples))), 0]])
            alpha = np.linalg.pinv(K) @ np.array(list(values)+[0])
            smoothed_signal = Lpinvm[:,samples]@alpha[:-1]+alpha[-1]
            return smoothed_signal

class Chain(Graph):
    def __init__(self, N, times=None):
        if times is None:
            self.times = np.ones((N,))
        else:
            self.times = times
        n_vertices = N
        edges = {1:[(2,1/(times[1]-times[0]))], N:[(N-1,1/(times[-1]-times[-2]))]}
        for i in range(2,N):
            n, l, r = i-1, i-2, i # 0-indexed list indices
            node, left, right = i, i-1, i+1 # corresponding 1-indexed node numbers
            edges[node] = [(left,1/(times[n]-times[l]) ), (right,1/(times[r]-times[n]) )]
        super().__init__(n_vertices, edges)
    def __str__(self):
        return super().__str__()


class Cycle(Graph):
    def __init__(self, N, times=None, mod=1):
        if times is None:
            self.times = np.ones((N,))
        else:
            self.times = times
        n_vertices = N
        edges = {1:[(2,1/(times[1]-times[0])), (N, 1/((times[-1]-times[0])-mod))], N:[(N-1,1/(times[-1]-times[-2])), (1, 1/((times[-1]-times[0])-mod))]}
        for i in range(2,N):
            n, l, r = i-1, i-2, i # 0-indexed list indices
            node, left, right = i, i-1, i+1 # corresponding 1-indexed node numbers
            edges[node] = [(left,1/(times[n]-times[l]) ), (right,1/(times[r]-times[n]) )]
        super().__init__(n_vertices, edges)
    def __str__(self):
        return super().__str__()



#### DCT-II ####

def get_dct_ii(n):
    X = np.zeros((n,n))
    for k in range(n):
        vec =np.array([np.cos(k*np.pi/n*(nn+0.5)) for nn in range(n)])
        scale = 1/np.sqrt(n) if k==0 else np.sqrt(2/n)
        X[:,k] = scale*vec
    return X

def get_L_dct_ii(n):
    L = circulant([2,-1]+[0]*(n-3)+[-1])
    L[0,:] = [1, -1]+[0]*(n-2)
    L[-1,:] = [0]*(n-2)+[-1,1]
    eigval, eigvec = np.linalg.eig(L)
    return L, eigvec

#### EXPERIMENTS ####

if __name__ == '__main__':
    number_nodes = 12
    dct_component = 4
    vis_p = -40
    m = 3
    sample_times = np.linspace(0,1,number_nodes+1)[:number_nodes]
    xs = np.arange(1,13)
    g = Cycle(number_nodes, times=sample_times,mod=sample_times[-2])
    test_function = np.array([415.14,415.79,416.17,416.36,415.74,416.56,416.61,416.45,416.88,417.27,417.03,417.43]) #get_L_dct_ii(number_nodes)[0][:,dct_component]

    L, A, D = g.get_laplacian_adjacency_degree_matrix()
    eigval, eigvec = np.linalg.eig(L)

    samples = np.arange(12)#[1,3,7,8,9]
    n_samples = len(samples)
    values = test_function[samples]

    plt.figure()
    plt.title(r"Spline smoothing in the chain graph $G$, $m=$"+f"{m}")
    plt.scatter(xs[samples], values, label='observations', marker='x', alpha=0.8)

    Lpinvm = np.linalg.pinv(L)**m
    K_g = Lpinvm[samples,:][:,samples]
    print("K_g", K_g)
    print("Lp^m", Lpinvm)
    for lambda_, color in zip([0.1, 0.01, 0.001, 0], ['royalblue', 'magenta', 'lime', 'orangered']):
        K = np.block([[K_g + lambda_*np.eye(n_samples), np.ones((n_samples,1))], [np.ones((1,n_samples)), 0]])
        alpha = np.linalg.pinv(K) @ np.array(list(values)+[0])
        interpolant = Lpinvm[:,samples]@alpha[:-1]+alpha[-1]
        plt.scatter(xs, interpolant, label=r"smoother, $\lambda=$"+f"{lambda_}", s=10, c=color)
        plt.plot(xs, interpolant, '--', alpha=0.3, c=color)
        plt.xlabel("node number (month)")
        plt.ylabel(r"CO$_2$ measurement (ppm)")
    plt.legend()
    plt.show()


    plt.figure()
    plt.title(r"Representers of evaluation in the chain graph $G$, $m=$"+f"{m}")
    for node, color in zip([5,0,6,11], ['royalblue', 'magenta', 'lime', 'orangered']):
        plt.scatter(xs, Lpinvm[:,node], label=f'representer of evaluation at node {node}', s=10, alpha=0.7, c=color)
        plt.plot(xs, Lpinvm[:,node], '--', c=color, alpha=0.2)
    plt.xlabel('node number')
    plt.legend()
    plt.show()



def solve(T, y, E, K_g, lambda_):
    pass


def upsample(B, ts, ts_up, reps, lambda_=0, m=2):
    n_samples, C = B.shape
    N = len(all_times)
    assert len(ts) == n_samples
    n_up = len(ts_up)
    all_times = sorted(list(set(ts+ts_up)))
    g = Chain(N, times=all_times)
    L, A, D = g.get_laplacian_adjacency_degree_matrix()
   
    Lpinvm = np.linalg.pinv(L)**m
 
    sample_locs = np.searchsorted(all_times, ts)
    K_g = Lpinvm[sample_locs,:][:,samples_locs]
    

    # unconstrained optimization to find v1
    K = np.block([[K_g + lambda_*np.eye(n_samples), np.ones((n_samples,1))], [np.ones((1,n_samples)), 0]])
    alpha = np.linalg.pinv(K) @ np.array(list(B[:,0])+[0])
    v1 = Lpinvm[:,sample_locs]@alpha[:-1]+alpha[-1]

    Vout = np.zeros((n_up, c))
    Vout[:,0] = v1


    # constrained optimization to find v2,...,vC
    rep_locs = np.searchsorted(all_times, reps)
    K_g = Lpinvm[rep_locs,:][:,rep_locs]
    for j in range(2, C+1):
        ij = j-1
        y = B[:,ij]
        T = Lpinvm[:,rep_locs]
        E = Vout[:,:ij].T@K    
        aj = solve(T, y, E, K_g, lambda_)
        Vout[:,i] = Lpinvm[:,sample_locs]@aj[:-1]+aj[-1]
    return Vout




@np.vectorize
def k1(x,u):
    return -u**3/6 + x*u**2/2 if u <= x else -x**3/6+u*x**2/2

def K(pts1, pts2):
    pts1, pts2 = np.meshgrid(pts1, pts2, indexing='ij')
    return k1(pts1, pts2)

def K_n(inducing_points):
    res = np.zeros((len(inducing_points)+2, len(inducing_points)+2))
    res[:-2, :-2] = K(inducing_points, inducing_points)
    return res

def K_s(sample_points, inducing_points):
    return K(sample_points, inducing_points)

def T(sample_points):
    M = len(sample_points)
    res = np.zeros((M,2))
    res[:,0] = 1
    res[:,1] = sample_points
    return res

def E(sample_points, inducing_points):
    return np.hstack([K_s(sample_points, inducing_points), T(sample_points)])

def O(alphas_solved, inducing_points):
    return alphas_solved @ W(inducing_points)

def W(inducing_points):
    n = len(inducing_points)
    @np.vectorize
    def kint(i,j):
        if i==n and j==n:
            return 1
        elif (i==n+1 and j==n) or (j==n+1 and i==n):
            return 0.5
        elif i==n or j==n:
            idx = min(i,j)
            pt = inducing_points[idx]
            return (pt**4-4*pt**3+6*pt**2)/24
        elif i==n+1 and j==n+1:
            return 1/3
        elif i==n+1 or j==n+1:
            idx = min(i,j)
            pt = inducing_points[idx]
            return (pt**5-10*pt**3+20*pt**2)/120
        else:
            idx_x = min(i,j)
            idx_y = max(i,j)
            x = inducing_points[idx_x]
            y = inducing_points[idx_y]
            return ( -x**7 + 7*x**6*y + x**3*(-35*y**4+140*y**3-210*y**2) + x**2*(21*y**5-210*y**3+420*y**2) )/5040
    is_, js_ = np.meshgrid(np.arange(n+2), np.arange(n+2), indexing='ij')
    return kint(is_, js_)
    
def upsample_l2_m2(B, ts, ts_ind, lambdas):
    M, C = B.shape
    m = 2
    sample_points = sorted(ts)
    inducing_points = sorted(list(set(ts).union(ts_ind)))
    n = len(inducing_points)
    E_ = E(ts, inducing_points)
    ETE = E_.T@E_
    Kn = K_n(inducing_points)
    Ws = W(inducing_points)
    K_ = np.block([[K(inducing_points, inducing_points)+M*lambdas[0]*np.eye(n),T(inducing_points)],[T(inducing_points).T, np.zeros((2,2))]])
    sample_locs = np.searchsorted(inducing_points, sample_points)
    u = np.zeros((m+n,1))
    u[sample_locs,0] = B[:,0]
    alphas_0 = np.linalg.pinv(K_)@ u
    alphas = alphas_0.copy().reshape((1,m+n))
    for j in range(1,C):
        Oj = alphas@Ws
        A = np.block([[2*(ETE + M*lambdas[j]*Kn), Oj.T], [Oj, np.zeros((j,j))]])
        b = np.vstack((2*E_.T@B[:,j].reshape((M,1)), np.zeros((j,1))))
        alpha = (np.linalg.pinv(A)@b)[:m+n]
        alphas = np.vstack((alphas, alpha.copy().reshape((1,m+n))))
    return alphas, inducing_points

def eval(alphas, inducing_points, eval_points):
    K_ = K(eval_points, inducing_points)
    val1 = K_@(alphas[:,:-2].T)
    T_ = T(eval_points)
    val2 = val1+T_@(alphas[:,-2:].T)
    return val2

nsample=12

B = get_dct_ii(nsample)[:,[2,7,8,11]]
M,C = B.shape
ts = np.linspace(0,1,nsample+1)[:nsample]
ts_ind = ts.copy() #np.linspace(0,1,93)
lambdas = [1e-7,1e-12,1e-12,1e-12]
Neval=nsample*100
alphas, inducing_points = upsample_l2_m2(B, ts, ts_ind, lambdas)
print("inducing points,", inducing_points)

eval_points = np.linspace(0,1,Neval+1)[:Neval]
evals = eval(alphas, inducing_points, eval_points)
plt.figure()
for i,c in zip(range(C), ['green', 'blue', 'red', 'orange']):
    plt.scatter(ts, B[:,i], c=c, marker='x')
    plt.plot(eval_points, evals[:,i], c=c, alpha=0.8)
plt.show()

print("alphas", alphas.shape, "W", W(inducing_points).shape)

print(alphas @ W(inducing_points) @ alphas.T)

eigval, eigvec = np.linalg.eig(W(inducing_points))
print("eigval", eigval)


def get_integrand(alphas, inducing_points, i, j):
    def f(x):
        K_ = K(np.array([x]), inducing_points)
        val1i = K_@(alphas[i,:-2].T)
        val1j = K_@(alphas[j,:-2].T)
        T_ = T(np.array([x]))
        return (val1i.flatten()+T_@(alphas[i,-2:].T).flatten())*(val1j+T_@(alphas[j,-2:].T).flatten())
    return f

for i in range(C):
    for j in range(C):
        print(f"L2 inner product between {i} and {j}", quad(get_integrand(alphas, inducing_points, i, j), 0, 1))

