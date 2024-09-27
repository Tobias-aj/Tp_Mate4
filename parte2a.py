import pandas as pd
import numpy as np

# Leer el archivo CSV usando pandas
df = pd.read_csv('players_21.csv')

# Establecemos value_eur como variable dependiente
y = df['value_eur'].values
n = len(y)

# Se crea una lista donde se almacenarán los indicadores de cada columna
skills = []

for col in df.select_dtypes(include=np.number).columns:
    if col != 'value_eur':
        X = df[col]
        if X.isnull().any():
            continue
        # B0 y B1 calculados por Mínimos Cuadrados:
        media_X = np.mean(X)
        media_Y = np.mean(y)
        aux1 = np.sum((X - media_X) * (y - media_Y))
        aux2 = np.sum((X - media_X) ** 2)

        if aux2 == 0:
            continue
        B1 = aux1 / aux2
        B0 = media_Y - (B1 * media_X)

        # Cálculo de R² (Coeficiente de Determinación) para cada X
        y_pred = B0 + B1 * X
        ss_total = np.sum((y - media_Y) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        R2 = 1 - ss_residual / ss_total

        # Cálculo de r (Coeficiente de Correlación) para cada X, Cálculo de Correlación de Pearson
        r = np.corrcoef(X, y)[0,1]

        skills.append({
            'name': col,
            'B0': B0,
            'B1': B1,
            'R2': R2,
            'r': r
        })

# Luego, se ordena la lista de diccionarios según valor de r, de mayor a menor
skills.sort(key=lambda x: x['r'], reverse=True)

print('\n\nCaracterísticas seleccionadas por r más significativo:\n')
for i in range(0,5):
    print(f"  Característica: {skills[i]['name']}")
    print(f"  Pendiente - B1: {skills[i]['B1']}")
    print(f"  Ordenada al origen - B0: {skills[i]['B0']}")
    print(f"  Coeficiente de determinación (R^2): {skills[i]['R2']}")
    print(f"  Coeficiente de correlación (r): {skills[i]['r']}")
    print("/" * 60)
print('\n\n\n')


# Seleccionar las columnas de interés (variables independientes) y confección de matriz
X = df[[skills[0]['name'], skills[1]['name'], skills[2]['name']]].values
x1 = df[skills[0]['name']]
x2 = df[skills[1]['name']]
x3 = df[skills[2]['name']]

x1_sum = x1.sum()
x2_sum = x2.sum()
x3_sum = x3.sum()
y_sum = y.sum()

x1_squared = x1.pow(2).sum()
x2_squared = x2.pow(2).sum()
x3_squared = x3.pow(2).sum()

#Construcción de las matrices X e Y
fila1 = np.array([n, x1_sum, x2_sum, x3_sum])
fila2 = np.array([x1_sum, x1_squared, x1_sum * x2_sum, x1_sum * x3_sum])
fila3 = np.array([x2_sum, x1_sum * x2_sum, x2_squared, x2_sum * x3_sum])
fila4 = np.array([x3_sum, x1_sum * x3_sum, x3_sum * x2_sum, x3_squared])

matrix_x = np.vstack((fila1, fila2))
matrix_x = np.round(matrix_x, decimals=5)
matrix_x = np.vstack((matrix_x, fila3))
matrix_x = np.round(matrix_x, decimals=5)
matrix_x = np.vstack((matrix_x, fila4))
matrix_x = np.round(matrix_x, decimals=5)

matrix_y = np.array([y_sum, y_sum * x1_sum, y_sum * x2_sum, y_sum * x3_sum])
matrix_y = np.round(matrix_y, decimals=5)


#Imprimir la matriz obtenida
print('Matriz inicial:')
print(matrix_x)

print(f'\nEl determinante de la matriz es: {np.linalg.det(matrix_x)}\n')

print('Matriz Y:')
print(matrix_y)
print(f'Y medio = {np.mean(matrix_y)}')


#Inversa de matrix_x
matrix_x_inv = np.linalg.inv(matrix_x)
matrix_x_inv = np.round(matrix_x_inv, decimals=20)
print('\nMatriz inversa de X:')
print(matrix_x_inv)

#Cálculo de los coeficientes B
B = np.dot(matrix_x_inv, matrix_y)
print('\nValores de coeficientes B:')
print(B)

ecuacion = ''
for i in range(len(B)):
        ecuacion += str(B[i]) if i == 0 else f' + {B[i]} · x{(i)}'
print('Ecuación de regresión lineal obtenida:')
print('**** ' + ecuacion + ' **** \n\n')


#################################################

# Indicadores

y_pred = []

for fila in X:
    x1,x2,x3 = fila
    pred = B[0] + B[1]*x1 + B[2]*x2 + B[3]*x3
    y_pred.append(pred)

error = [y[i] - y_pred[i] for i in range(0, n)]

SCE = 0
SCR = 0
for i in range(0,n):
     SCE += (error[i] ** 2)
     SCR += ((y_pred[i] - media_Y) ** 2)
STC = SCE + SCR
error_std = np.sqrt(SCE/(n - (3 + 1)))
print(f'Error Estándar: {error_std}\n\n')

R2 = SCR / STC
print(f'R²: {R2}')

R2_ajustado = 1 - (1 - R2) * ( (n - 1) / (n - 3 - 1))
print(f'R² ajustado: {R2_ajustado}')

r = np.sqrt(R2)
print(f'Coeficiente de Correlación r: {r}')