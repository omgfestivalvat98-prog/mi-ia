import random
import math

# --- Datos de entrenamiento: XOR ---
datos = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

# --- Inicializar pesos aleatorios ---
random.seed(42)
w1 = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
b1 = [random.uniform(-1, 1) for _ in range(2)]
w2 = [random.uniform(-1, 1) for _ in range(2)]
b2 = random.uniform(-1, 1)

def sigmoide(x):
    return 1 / (1 + math.exp(-x))

def forward(x):
    h = []
    for i in range(2):
        z = w1[i][0]*x[0] + w1[i][1]*x[1] + b1[i]
        h.append(sigmoide(z))
    z = w2[0]*h[0] + w2[1]*h[1] + b2
    return sigmoide(z), h

def entrenar(epocas=10000, lr=0.5):
    global w1, b1, w2, b2
    for epoca in range(epocas):
        for x, y_real in datos:
            y_pred, h = forward(x)
            error = y_pred - y_real

            d2 = error * y_pred * (1 - y_pred)
            d1 = []
            for i in range(2):
                d1.append(d2 * w2[i] * h[i] * (1 - h[i]))

            for i in range(2):
                w2[i] -= lr * d2 * h[i]
            b2 -= lr * d2

            for i in range(2):
                for j in range(2):
                    w1[i][j] -= lr * d1[i] * x[j]
                b1[i] -= lr * d1[i]

        if epoca % 2000 == 0:
            perdida = sum((forward(x)[0] - y)**2 for x, y in datos) / len(datos)
            print(f"Epoca {epoca:5d} | Perdida: {perdida:.6f}")

entrenar()

print("\n--- Resultados ---")
for x, y in datos:
    pred, _ = forward(x)
    print(f"Entrada {x} -> Prediccion {pred:.4f} (esperado {y})")

