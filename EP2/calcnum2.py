import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import ListedColormap

# Defina os parâmetros
epsilon = 1e-6
kmax = 20
N = 100  # Tamanho da grade

# Defina as soluções conhecidas do sistema
solutions = [(1, 0), (-1, 0)]

# Crie uma matriz para armazenar as cores das regiões de convergência
colors_convergence = np.zeros((N+1, N+1), dtype=int)

# Crie uma matriz para armazenar o número de iterações
iterations = np.zeros((N+1, N+1), dtype=int)

# Crie uma paleta de cores para as soluções
colormap = ListedColormap(['r', 'g'])

# Função para o sistema de equações
def f(x, y):
    return np.array([x**2 - y**2 - 1, 2 * x * y])

# Função para calcular a matriz jacobiana de f(x, y)
def jacobian(x, y):
    return np.array([[2*x, -2*y], [2*y, 2*x]])

# Loop para percorrer os pontos na grade T
for i in range(N+1):
    for j in range(N+1):
        x0 = -1 + (2 * i / N)
        y0 = -1 + (2 * j / N)
        x = x0
        y = y0
        converged = False

        for k in range(kmax):
            jacobian_matrix = jacobian(x, y)
            if np.linalg.cond(jacobian_matrix) < 1 / sys.float_info.epsilon:
                delta = np.linalg.solve(jacobian_matrix, -f(x, y))
                x += delta[0]
                y += delta[1]
                if np.linalg.norm(delta) < epsilon:
                    converged = True
                    break
            else:
                break

        # Verifique para qual solução (se alguma) o método convergiu
        dif_menor = 10000000
        for idx, sol in enumerate(solutions):
            dif = np.linalg.norm([x, y] - np.array(sol))
            if  dif < epsilon and dif < dif_menor:
                dif_menor = dif
                colors_convergence[j, i] = idx # Cor 1 ou 2
                print(round(x0,3), round(y0,3), " sol = ", idx, " its = ",k+1)
                #break

        iterations[j, i] = k + 1 if converged else kmax  # Número de iterações

# Crie o mapa de cores das regiões de convergência
plt.figure(figsize=(10, 5))
plt.imshow(colors_convergence, extent=(-1, 1, -1, 1), cmap=colormap)
plt.colorbar(ticks=[1, 2], label='Solutions (1 or 2)')
plt.title('Convergence Regions')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Crie o mapa de cores do número de iterações
plt.figure(figsize=(10, 5))
plt.imshow(iterations, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
plt.colorbar(label='Iterations')
plt.title('Number of Iterations')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
