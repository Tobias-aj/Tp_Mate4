import pandas as pd
import numpy as np

# Cargar los datos
data = pd.read_csv('players_21.csv')

# Seleccionamos las características específicas: 'wage_eur', 'overall', 'potential'
X = data[['wage_eur', 'overall', 'potential']].fillna(0)  # Reemplazar NaN con 0
y = data['value_eur']

# Normalizar las características (opcional, pero recomendado para el descenso por gradiente)
X = (X - X.mean()) / X.std()

# Añadir una columna de unos a X para el término de intercepto
X = np.c_[np.ones(X.shape[0]), X]  # Añadimos la columna de unos (para B0)

# Inicializar parámetros para descenso por gradiente
aprendizaje = 0.10
n_iteraciones = 2000
m = len(y)

# Inicializar los coeficientes (B)
B = np.zeros(X.shape[1])

# Variables para el criterio de corte basado en la convergencia del ECM
mse_anterior = np.inf  # Inicializamos con un valor grande
umbral = 1e-6  # Umbral para detener el descenso por gradiente si el ECM cambia muy poco

# Algoritmo de descenso por gradiente
for iteration in range(n_iteraciones):
    # Predicción
    y_pred = np.dot(X, B)
    
    # Gradiente de la función de costo respecto a B
    gradient = (-2/m) * np.dot(X.T, (y - y_pred))
    
    # Actualización de los coeficientes B
    B = B - aprendizaje * gradient
    
    # Calculamos el ECM actual
    mse_actual = np.mean((y - y_pred) ** 2)
    

##Importante si se desea hacer el algoritmo con metodo de iteraciones, comentar la linea de abajo#########

    # Verificamos el cambio en el ECM para criterio de corte
    if abs(mse_anterior - mse_actual) < umbral:
        print(f"El algoritmo ha convergido en la iteración {iteration} con ECM = {mse_actual}")
        break
    
    mse_anterior = mse_actual
    
    # Cada 100 iteraciones mostramos el ECM
    if iteration % 100 == 0:
        print(f"Iteración {iteration}: ECM = {mse_actual}")

print("$" * 50)

# Resultados usando ecuaciones normales
B_normales = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Resultados finales después de descenso por gradiente
print("Resultados finales después de descenso por gradiente:")

print(f"Intercepto (B0): {B[0]}")
print(f"Coeficiente para 'wage_eur' (B1): {B[1]}")
print(f"Coeficiente para 'overall' (B2): {B[2]}")
print(f"Coeficiente para 'potential' (B3): {B[3]}")
print("-" * 50)

# Resultados usando ecuaciones normales
print("Resultados usando ecuaciones normales:")

print(f"Intercepto (B0): {B_normales[0]}")
print(f"Coeficiente para 'wage_eur' (B1): {B_normales[1]}")
print(f"Coeficiente para 'overall' (B2): {B_normales[2]}")
print(f"Coeficiente para 'potential' (B3): {B_normales[3]}")
