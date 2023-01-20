import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
samples = np.array([2,3,6])
values = np.array([4+np.random.randn(), 6+np.random.randn(), 12+np.random.randn()])
interpolant = np.zeros((10,))

plt.figure()
plt.title(r"Interpolation in $\mathbb{R}^{10}$ is boring without graphs")
plt.scatter(samples, values, label='observations', marker='x')
for lambda_ in [1,0.01, 0]:
	interpolant[samples] = values*1/(1+lambda_)
	plt.scatter(x, interpolant, label=r"interpolant, $\lambda=$"+f"{lambda_}", alpha=0.5, s=5)
plt.legend()
plt.show()
