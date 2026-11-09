import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA
# ==========================================
fs = 1000                     # Frecuencia de muestreo (Hz)
t = np.linspace(0, 1, fs, endpoint=False) # Vector de tiempo de 1 segundo

# Señal compuesta: Onda de interés (5 Hz) + componente de alta frecuencia (100 Hz)
f_senal = 5
f_ruido_alto = 100
senal_pura = np.sin(2 * np.pi * f_senal * t)
interferencia = 0.5 * np.sin(2 * np.pi * f_ruido_alto * t)

# Adición de ruido blanco gaussiano para simular ruido ambiental
np.random.seed(42)
ruido_blanco = 0.2 * np.random.normal(size=t.shape)

# Señal total de entrada (Sucia)
senal_entrada = senal_pura + interferencia + ruido_blanco

# ==========================================
# 2. DISEÑO DEL FILTRO DIGITAL (IIR Butterworth Pasa Bajos)
# ==========================================
orden = 4                     # Orden del filtro
f_corte = 20                  # Frecuencia de corte (Hz) -> Permite pasar < 20 Hz
nyquist = 0.5 * fs            # Frecuencia de Nyquist
cutoff_normalizado = f_corte / nyquist

# Diseño del filtro Butterworth pasa bajos
b, a = signal.butter(orden, cutoff_normalizado, btype='low', analog=False)

# ==========================================
# 3. APLICACIÓN DEL FILTRO A LA SEÑAL
# ==========================================
# Se utiliza filtfilt para aplicar filtrado de fase cero (evita desfases temporales)
senal_filtrada = signal.filtfilt(b, a, senal_entrada)

# Cálculo de la respuesta en frecuencia del filtro
w, h = signal.freqz(b, a, worN=8000)
frecuencias_hz = (w / np.pi) * nyquist

# ==========================================
# 4. VISUALIZACIÓN DE LOS RESULTADOS
# ==========================================
plt.figure(figsize=(12, 10))

# Gráfica 1: Señal de entrada con ruido
plt.subplot(3, 1, 1)
plt.plot(t, senal_entrada, color='gray', alpha=0.7, label='Señal con Ruido')
plt.plot(t, senal_pura, color='blue', linestyle='--', label='Señal Pura Original')
plt.title('1. Definición de la Señal de Entrada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

# Gráfica 2: Respuesta en frecuencia del filtro
plt.subplot(3, 1, 2)
plt.plot(frecuencias_hz, np.abs(h), color='purple', linewidth=2)
plt.title('2. Respuesta en Frecuencia (Filtro Butterworth Pasa Bajos)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (Magnitud)')
plt.axvline(f_corte, color='red', linestyle=':', label=f'Frecuencia de Corte ({f_corte} Hz)')
plt.legend(loc='upper right')
plt.grid(True)

# Gráfica 3: Señal después de aplicar el filtro
plt.subplot(3, 1, 3)
plt.plot(t, senal_filtrada, color='green', linewidth=1.5, label='Señal Filtrada (Salida)')
plt.plot(t, senal_pura, color='blue', linestyle='--', alpha=0.5, label='Señal Pura Original')
plt.title('3. Aplicación del Filtro y Comparación')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()