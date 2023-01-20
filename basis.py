from scipy.linalg import circulant, sqrtm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import jv
import cvxpy as cp

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

def eval_(alphas, inducing_points, eval_points):
    K_ = K(eval_points, inducing_points)
    print("K", K_.shape, "alphas", alphas.shape)
    val1 = K_@(alphas[:-2].T)
    T_ = T(eval_points)
    val2 = val1+T_@(alphas[-2:].T)
    return val2

def eval_2(inducing_points):
    n = len(inducing_points)
    E = np.zeros((5,n+2))
    E[0,-2] = 1
    E[0,-1] = 0.5
    E[1,-1] = np.sqrt(3)/6
    for i,x in enumerate(inducing_points):
        E[0,i] = x**4/24 - x**3/6 + x**2/4
        E[1,i] = 2*np.sqrt(3)*(x**5/120 - x**4/48 + x**2/24)
        E[2,i] = 6*np.sqrt(5)*(x**6/360 - x**5/120 + x**4/144)
        E[3,i] = 20*np.sqrt(7)*(x**7/840-x**6/240+x**5/200-x**4/480)
        E[4,i] = 210*(x**8/1680-x**7/420+x**6/280-x**5/420+x**4/1680)
    return E

def eval_x(inducing_points,r=5):
    n = len(inducing_points)
    E = np.zeros((r+1, n+2))
    E[0,-2] = 1
    E[0,-1] = 0.5
    for i,x in enumerate(inducing_points):
        E[0,i] = x**4/24 - x**3/6 + x**2/4
        for l in range(1,r+1):
           E[l,i] = (np.exp(l*x)-1-l*x)/(l**4) + x**2*np.exp(l)*(3*l-x-3)/(6*l**2)
           if i==0:
               E[l,1] =  (np.exp(l)*(l-1)+1)/(l**2)
               E[l,0] = (np.exp(l)-1)/l
    return E

def phi_0(x):
    return 1
def phi_1(x):
    return 2*np.sqrt(3)*(x-1/2)
def phi_2(x):
    return 6*np.sqrt(5)*(x**2-x+1/6)
def phi_3(x):
    return 20*np.sqrt(7)*(x**3-3*x**2/2+3*x/5-1/20)
def phi_4(x):
    return 210*(x**4-2*x**3+9*x**2/7-2*x/7+1/70)

test_fn = lambda x: jv(2.5,x)


vals = [quad(test_fn,0,1)[0],
        quad(lambda x: test_fn(x)*np.exp(x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(2*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(3*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(4*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(5*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(6*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(7*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(8*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(9*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(10*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(11*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(12*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(13*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(14*x), 0, 1)[0],
        quad(lambda x: test_fn(x)*np.exp(15*x), 0, 1)[0],
]

n_evals = 16
vals = vals[:n_evals]
n_points = 150
inducing_points = np.linspace(0,1,n_points+2)[1:n_points+1]
E = eval_x(inducing_points,n_evals-1)[:n_evals,:]
x = cp.Variable(n_points+3)
f = np.zeros(n_points+3)
f[-1]=1
Ws = W(inducing_points)
print(np.linalg.eig(Ws)[0])
print("testing Ws symmetric...")
print(np.allclose(Ws, Ws.T, rtol=1e-10, atol=1e-10))
print(np.linalg.eig(Ws)[0])
eigval, eigvec = np.linalg.eig(Ws)
rW = eigvec@np.diag(np.sqrt(np.where(eigval >= 1e-10, eigval, 0)))@eigvec.T
Ks = K_n(inducing_points)
eigval, eigvec = np.linalg.eig(Ks)
rK = eigvec@np.diag(np.sqrt(np.where(eigval >= 1e-10, eigval, 0)))@eigvec.T
soc_constraints = [#cp.norm((rW)@(x[:-1]),2) <= np.ones((n_points+2,))+x[-1],
                   cp.norm(rK@(x[:-1]),2) <= x[-1]] 
print("Ws", Ws)
print("rW", rW.shape, "E", E.shape)
print("E", E)
print("rW", rW)
prob = cp.Problem(cp.Minimize(f@x), soc_constraints+[E@(x[:-1]) == np.array(vals)])
prob.solve()
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)
for i in range(1):
    print("SOC constraint %i dual variable solution" % i)
    print(soc_constraints[i].dual_value)
print("inducing_points", len(inducing_points), "x", x.value)


eval_points = np.linspace(0,1,1000)
plt.figure()
plt.plot(eval_points, eval_((x.value)[:-1], inducing_points, eval_points), label='reconstruction')
plt.plot(eval_points, test_fn(eval_points), label=r'$J_{2.5}$')
plt.xlabel(r"$x$")
plt.title(r"Reconstruction based on $v_0,\ldots,v_8$,"+f" {n_points} inducing points")
#phi_0(eval_points)+phi_1(eval_points)+phi_2(eval_points)+1*phi_3(eval_points)+0*phi_4(eval_points), label=r'$\phi_4$')
plt.legend()
plt.show()
