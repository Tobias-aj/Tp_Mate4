import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt

# Leer el archivo CSV usando pandas
df = pd.read_csv('players_21.csv')

# Seleccionamos las columnas de interés
X = df['overall']
y = df['value_eur']

X = X.tolist()
y = y.tolist()

# Calcular valores necesarios para la regresión
n = len(X)
promedio_X = sum(X) / n
promedio_y = sum(y) / n

S_xy = sum((X[i] - promedio_X) * (y[i] - promedio_y) for i in range(n))
S_xx = sum((X[i] - promedio_X) ** 2 for i in range(n))
S_yy = sum((y[i] - promedio_y) ** 2 for i in range(n))

# Pendiente y ordenada
b1 = S_xy / S_xx
b0 = promedio_y - b1 * promedio_X

# Predicción de y basada en la regresión
y_pred = [b0 + b1 * X[i] for i in range(n)]

# Suma de los cuadrados de los residuos
SS_r = sum((y[i] - y_pred[i]) ** 2 for i in range(n))

# Coeficiente de determinación (R²)
R_cuadrado = 1 - (SS_r / S_yy)

# Correlación lineal (r)
r = S_xy / ((S_xx * S_yy) ** 0.5)

# Desviación estándar de los residuos (S)
S = math.sqrt(SS_r / (n - 2))

# Estadístico T para la pendiente (b1)
theta = 0
Ts = (b1 - theta) / (S / math.sqrt(S_xx))

# Grados de libertad
Gl = n - 2

# Valor crítico de t
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha / 2, Gl)

# Estimador insesgado de a²
estimadorA = S ** 2

# Intervalo de confianza para β1
IC_b1_inf = b1 - t_crit * math.sqrt(estimadorA / S_xx)
IC_b1_sup = b1 + t_crit * math.sqrt(estimadorA / S_xx)

# Imprimir resultados
print(f"Pendiente (β1): {b1}")
print(f"Ordenada al origen (β0): {b0}")
print(f"Coeficiente de determinación (R²): {R_cuadrado}")
print(f"Correlación lineal (r): {r}")
print(f"Desviación estándar de los residuos (S): {S}")
print(f"T: {Ts}")
print(f"Valor crítico de t: {t_crit}")

if abs(Ts) > t_crit:
    print("Rechazamos la hipótesis nula. La pendiente es significativamente diferente de 0.")
else:
    print("No podemos rechazar la hipótesis nula. No hay evidencia suficiente para decir que la pendiente es diferente de 0.")
print(f"Intervalo de confianza para β1: [{IC_b1_inf}, {IC_b1_sup}]")

# Graficar los puntos y la línea de regresión
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Datos observados')  # Puntos originales
plt.plot(X, y_pred, color='red', label='Línea de regresión')  # Línea de regresión
plt.xlabel('Overall (Calificación general)')
plt.ylabel('Value (Valor en euros)')
plt.title('Regresión lineal: Valor vs Overall')
plt.legend()
plt.grid(True)
plt.show()
