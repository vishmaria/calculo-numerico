import numpy as np

def gauss_seidel(N, max_iterations, tolerance):
    # Inicialize a matriz de temperaturas
    T = np.zeros((N, N))
    for i in range(N):
        T[i][N-1] = 128
    
    for k in range(max_iterations):
        max_diff = 0.0
        
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                T_old = T[i, j]
                T[i, j] = 0.25 * (T[i-1, j] + T[i+1, j] + T[i, j-1] + T[i, j+1]) #prob2 ex1
                diff = abs(T[i, j] - T_old)
                max_diff = max(max_diff, diff)
        
        if max_diff < tolerance:
            print(f"Convergência alcançada após {k+1} iterações.")
            break

    return T

# Parâmetros
N_values = [10, 25, 50, 100]
max_iterations = 10000
tolerance = 1e-2

for N in N_values:
    T = gauss_seidel(N, max_iterations, tolerance)
    print(f"Para N = {N}, Temperatura em T(0.5, 0.5) é aproximadamente {T[N//2, N//2]:.6f}")
