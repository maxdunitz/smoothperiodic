import numpy as np
import matplotlib.pyplot as plt

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

class Line(Graph):
	def __init__(self, N):
		n_vertices = N
		edges = {}
		for i in range(N):
			node = i+1
			left = (node-2)%N+1
			right = (node)%N+1
			edges[node] = [(left,1), (right,1)]
		super().__init__(n_vertices, edges)
	def __str__(self):
		return super().__str__()

class LineEnd(Graph):
	def __init__(self, N):
		n_vertices = N
		edges = {1:[(N,1)], N:[(1,1)]}
		super().__init__(n_vertices, edges)
	def __str__(self):
		return super().__str__()

class Image(Graph):
	def __init__(self, M, N):
		n_vertices = M*N
		edges = {}
		for i in range(M):
			for j in range(N):
				node = i*M+j+1
				left = i*M+(j-1)%N+1
				right = i*M+(j+1)%N+1
				top = ((i-1)%M)*M+j+1
				bottom = ((i+1)%M)*M+j+1
				edges[node] = [(left,1), (right,1), (top,1), (bottom,1)]
		super().__init__(n_vertices, edges)


if __name__ == '__main__':
	g = Line(10)
	print(g)
	L, A, D = g.get_laplacian_adjacency_degree_matrix()
	print(A)
	eigval, eigvec = np.linalg.eig(L)
	plt.figure()
	for i in range(10):
		if i%5 == 0:
			plt.plot(np.arange(1,11), eigvec[:,i], alpha=0.5, label=f"{20-i}, eigval {eigval[i]}")
	plt.legend()
	plt.show()

	g2 = LineEnd(10)
	L2, A2, D2 = g2.get_laplacian_adjacency_degree_matrix()
	print(A2)
	print(D2)
	print("L", L-L2)
	eigval, eigvec = np.linalg.eig(L2)
	plt.figure()
	for i in range(10):
		print("i", i, eigval[i], np.abs(eigvec[i,0]-eigvec[i,-1]))
		if i == 0:
			plt.plot(np.arange(1,11), eigvec[:,i], alpha=0.5, label=f"{20-i}, eigval {eigval[i]}")
	plt.legend()
	plt.show()

	x = np.arange(10)
	samples = np.array([2,3,6])
	values = np.array([4+np.random.randn(), 6+np.random.randn(), 12+np.random.randn()])
	interpolant = np.zeros((10,))

	plt.figure()
	plt.title(r"Interpolation in $\mathbb{R}^{10}$ with graphs")
	plt.scatter(samples, values, label='observations', marker='x')
	for lambda_ in [1, 0]:
		L = L-L2
		K = np.linalg.pinv(L)[samples, samples]
		alpha = np.linalg.pinv(K + lambda_*np.eye(3)) @ values
		interpolant = np.linalg.pinv(L)[:,samples]@alpha
		plt.scatter(x, interpolant, label=r"interpolant, $\lambda=$"+f"{lambda_}", alpha=0.5, s=5)
	plt.legend()
	plt.show()




