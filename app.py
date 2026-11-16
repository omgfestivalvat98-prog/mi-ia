import random
import math

# --- Entrenamiento base ---
datos = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
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

def entrenar(epocas=5000, lr=0.5):
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

print("Entrenando a la IA... (espera unos segundos)")
entrenar()
print("¡IA lista para conversar!\n")

while True:
    entrada = input("Escribe dos numeros (0 o 1) separados por espacio, o escribe 'salir': ")
    
    if entrada.lower() == 'salir':
        print("¡Adiós! La IA se apaga.")
        break
        
    try:
        partes = entrada.split()
        if len(partes) != 2:
            print("❌ Error: Debes escribir exactamente dos numeros (ej: 0 1)\n")
            continue
            
        x_usuario = [int(partes[0]), int(partes[1])]
        
        if x_usuario[0] not in [0, 1] or x_usuario[1] not in [0, 1]:
            print("❌ Error: Solo puedes usar 0 o 1.\n")
            continue
            
        prediccion, _ = forward(x_usuario)
        resultado_final = 1 if prediccion > 0.5 else 0
        
        print(f"🤖 La IA calcula: {prediccion:.4f} -> Resultado final: {resultado_final}\n")
        
    except ValueError:
        print("❌ Error: Escribe solo numeros enteros.\n")

