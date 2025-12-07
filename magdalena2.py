#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import time
import threading
import re
from datetime import datetime, timedelta
import sounddevice as sd
import numpy as np
import wave
import asyncio
import tempfile
from playsound import playsound
import edge_tts

from dotenv import load_dotenv


# Se asume que RPi.GPIO está instalado en el entorno de ejecución
try:
    import RPi.GPIO as GPIO
    GPIO_INSTALADO = True
except ImportError:
    print("Advertencia: RPi.GPIO no encontrado. El monitoreo de botones GPIO estará inactivo.")
    GPIO_INSTALADO = False

# Se asume que faster_whisper está instalado
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster_whisper no encontrado. Instálalo para la transcripción.")
    exit()

# Se asume que la biblioteca huggingface_hub está instalada
try:
    from huggingface_hub import InferenceClient
except ImportError:
    print("Error: huggingface_hub no encontrado. Instálalo para el procesamiento de tareas.")
    exit()

# --------------------------
# Configuración y Constantes
# --------------------------
ARCHIVO_ACCIONES = "acciones.json"  # Corresponde a 'tareas.json' en el segundo código
ARCHIVO_HISTORIAL = "acciones_historial.json"  # Corresponde a 'historico.json' en el segundo código
PIN_ROJO = 17
PIN_VERDE = 27
FS = 16000  # frecuencia de muestreo
DEBOUNCE = 0.3
BUFFER_MINUTES = 5 # Margen de conflicto de horario

# Configuración de GPIO
if GPIO_INSTALADO:
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_ROJO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(PIN_VERDE, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    except Exception as e:
        print(f"Error al configurar GPIO: {e}")
        GPIO_INSTALADO = False


# --------------------------
# Modelo Whisper y Hugging Face
# --------------------------
print("Cargando modelo Whisper...")
try:
    whisper_model = WhisperModel("base", device="cpu")
except Exception as e:
    print(f"Error al cargar el modelo Whisper: {e}")
    whisper_model = None


# Cargar las variables del .env
load_dotenv()

# Configuración de Hugging Face
HF_API_KEY = os.getenv("HF_API_KEY")
LLAMA_MODEL = os.getenv("LLAMA_MODEL")

try:
    client = InferenceClient(api_key=HF_API_KEY)
except Exception as e:
    print(f"Error al inicializar InferenceClient: {e}")
    client = None

# =========================
# CONTEXTOS PARA LLAMA
# =========================
CONTEXTO_TAREAS = """
Eres un sistema para la gestión de tareas diarias para personas mayores con problemas leves de memoria.

Tu trabajo es convertir mensajes naturales en una lista de tareas con horario.

FORMATO:
- Devuelve SOLO un JSON válido con una lista de objetos { "tarea": ..., "horario": ... }
- Si no entiendes el mensaje → devuelve exactamente: ERROR
- NO añadas nada más fuera del JSON o ERROR.

Ejemplos:
Entrada: "Voy a desayunar a las 9 y caminar a las 10"
Salida:
[
  {"tarea":"desayunar","horario":"09:00"},
  {"tarea":"caminar","horario":"10:00"}
]

Entrada: "asdf 123"
Salida:
ERROR
"""

CONTEXTO_NARRACION = """
Eres un narrador de voz amable y conciso para un sistema de gestión de tareas diarias para personas mayores.

Tu tarea es reformular un mensaje de aviso sobre tareas no guardadas debido a conflictos de horario. El texto debe ser fácil de entender, tranquilizador y optimizado para una voz sintética.

Instrucciones:
1. Siempre empieza con una frase de aviso amable (ej. "Atención, se detectaron algunos conflictos...").
2. No uses formato de lista, solo texto corrido o frases cortas.
3. Menciona la tarea que se intentó añadir, su hora, la tarea que causa el conflicto y su hora.
4. No uses emojis, solo texto.
"""

# --------------------------
# Variables de estado
# --------------------------
grabando = False
alerta_activa = False
detener_alerta = False

# --------------------------
# Funciones JSON (Adaptadas del segundo código)
# --------------------------

def load_json(path, default):
    """Carga un archivo JSON, devuelve el valor por defecto si hay un error o no existe."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except Exception as e:
        print(f"Error al cargar {path}: {e}")
        return default


def save_json(path, data):
    """Guarda datos en un archivo JSON."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error al guardar {path}: {e}")

# Funciones del primer código re-mapeadas a las nuevas rutas (para compatibilidad con el resto del código original)
def cargar_acciones():
    """Carga acciones/tareas (ARCHIVO_ACCIONES)."""
    return load_json(ARCHIVO_ACCIONES, [])

def guardar_acciones(acciones):
    """Guarda acciones/tareas (ARCHIVO_ACCIONES)."""
    save_json(ARCHIVO_ACCIONES, acciones)

def cargar_historial():
    """Carga el historial (ARCHIVO_HISTORIAL)."""
    return load_json(ARCHIVO_HISTORIAL, [])

def guardar_historial(historial):
    """Guarda el historial (ARCHIVO_HISTORIAL)."""
    save_json(ARCHIVO_HISTORIAL, historial)

# --------------------------
# Funciones de Tiempo y Utilidad
# --------------------------

def time_to_datetime(time_str):
    """Convierte 'HH:MM' a un objeto datetime.datetime (con fecha de hoy)"""
    today = datetime.now().date()
    try:
        hour, minute = map(int, time_str.split(':'))
        return datetime(today.year, today.month, today.day, hour, minute, 0)
    except ValueError:
        return None 

def is_time_in_range(target_time_dt, existing_time_dt, buffer_minutes=BUFFER_MINUTES):
    """
    Comprueba si target_time_dt está dentro del margen de existing_time_dt.
    target_time_dt está en conflicto si está entre [existente - buffer, existente + buffer].
    """
    if not target_time_dt or not existing_time_dt:
        return False
        
    delta_buffer = timedelta(minutes=buffer_minutes)
    
    # Rango de tiempo conflictivo para la tarea existente
    lower_bound = existing_time_dt - delta_buffer
    upper_bound = existing_time_dt + delta_buffer
    
    return lower_bound <= target_time_dt <= upper_bound

def obtener_proxima_accion():
    """Devuelve la próxima acción programada (primer código)."""
    acciones = cargar_acciones()
    ahora = datetime.now()
    futuras = []
    for acc in acciones:
        try:
            # Asegurar que las tareas que ya tienen fecha en el historial no interfieran si se cargan por error.
            if "fecha" in acc: continue 

            hora_accion = datetime.strptime(acc["horario"], "%H:%M").replace(
                year=ahora.year, month=ahora.month, day=ahora.day
            )
            if hora_accion >= ahora:
                futuras.append((hora_accion, acc["tarea"])) # Usar 'tarea' del segundo código
        except:
            continue
    if not futuras:
        return None
    futuras.sort(key=lambda x: x[0])
    return futuras[0]

# --------------------------
# Funciones de Audio y TTS
# --------------------------

def grabar_mientras_pulsado_rojo(nombre_archivo="grabacion_rojo.wav", fs=FS):
    """Graba audio mientras se pulsa el botón rojo (primer código)."""
    global grabando
    if not GPIO_INSTALADO or grabando:
        return None # No intenta grabar si no hay GPIO o ya está grabando
        
    grabando = True
    print("Mantén pulsado el botón rojo para grabar")
    audio_buffer = []

    def callback(indata, frames, time, status):
        audio_buffer.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
        while GPIO.input(PIN_ROJO) == 0:
            sd.sleep(50)

    grabando = False
    
    if audio_buffer:
        audio = np.concatenate(audio_buffer, axis=0)
        with wave.open(nombre_archivo, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio.tobytes())
        print(f"Grabación guardada en {nombre_archivo}")
        return nombre_archivo
    return None

def transcribir_audio(nombre_archivo):
    """Transcribe el audio usando Whisper (primer código)."""
    if not whisper_model:
        return "ERROR DE TRANSCRIPCIÓN"
        
    print("Transcribiendo audio...")
    try:
        segments, _ = whisper_model.transcribe(nombre_archivo, beam_size=5)
        texto = " ".join([seg.text for seg in segments])
        print("Dijo:", texto)
        return texto.strip()
    except Exception as e:
        print(f"Error en la transcripción: {e}")
        return "ERROR DE TRANSCRIPCIÓN"

def reproducir_texto(texto):
    """Reproduce el texto usando Edge TTS (primer código)."""
    async def tts():
        try:
            tts_obj = edge_tts.Communicate(texto, voice="es-ES-ElviraNeural")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                temp_path = f.name
            await tts_obj.save(temp_path)
            playsound(temp_path)
            os.remove(temp_path)
        except Exception as e:
            print("Error TTS:", e)

    threading.Thread(target=lambda: asyncio.run(tts()), daemon=True).start()

# --------------------------
# Lógica de Hugging Face y Tareas
# --------------------------

def narrar_conflicto(tareas_rechazadas):
    """
    Crea el mensaje crudo de conflicto y lo narra usando el modelo de Llama.
    Devuelve la versión optimizada para voz (segundo código).
    """
    if not client:
        return "Error en el sistema de narración. Revisa la clave API o el cliente."

    # 1. Crear el mensaje de conflicto en formato lista (Input para el modelo)
    mensaje_conflicto_crudo = "Las siguientes tareas no han sido guardadas porque caen muy cerca (±5 minutos) de otra tarea:\n"
    for rechazo in tareas_rechazadas:
        mensaje_conflicto_crudo += f"- La tarea '{rechazo['tarea_nueva']}' a las {rechazo['horario']} se solapa con '{rechazo['conflicto_con']}' a las {rechazo['horario_conflicto']}.\n"

    print("\n--- Texto de Conflicto CRUDO (Input al modelo de Narración) ---\n")
    print(mensaje_conflicto_crudo)
    print("--------------------------------------------------------------\n")
    
    # 2. Enviar a Hugging Face para la narración
    messages_narracion = [
        {"role": "system", "content": CONTEXTO_NARRACION},
        {"role": "user", "content": mensaje_conflicto_crudo}
    ]
    
    try:
        completion = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=messages_narracion
        )
        
        respuesta_narrada = completion.choices[0].message.content.strip()
        return respuesta_narrada
        
    except Exception as e:
        print(f"Error al llamar al modelo de narración: {e}")
        return mensaje_conflicto_crudo


def merge_tasks(current_tasks, new_tasks, buffer_minutes=BUFFER_MINUTES):
    """
    Agrega tareas nuevas a `current_tasks`. 
    Rechaza si cae dentro de los +/- buffer_minutes de una existente (segundo código).
    """
    
    horarios_existentes_dt = []
    # Usar 'tarea' y 'horario' para mantener la compatibilidad con el nuevo formato
    for t in current_tasks:
        dt = time_to_datetime(t["horario"])
        if dt:
            horarios_existentes_dt.append({
                "dt": dt,
                "horario": t["horario"],
                "tarea": t["tarea"]
            })
            
    tareas_aceptadas = []
    tareas_rechazadas = []

    for t in new_tasks:
        horario_nuevo = t["horario"]
        tarea_nueva = t["tarea"]
        dt_nuevo = time_to_datetime(horario_nuevo)
        
        if not dt_nuevo:
            continue

        conflicto = None
        for existente in horarios_existentes_dt:
            if is_time_in_range(dt_nuevo, existente["dt"], buffer_minutes):
                conflicto = existente
                break
        
        if not conflicto:
            # Tarea aceptada: añadir a la lista principal
            current_tasks.append(t)
            horarios_existentes_dt.append({"dt": dt_nuevo, "horario": horario_nuevo, "tarea": tarea_nueva})
            tareas_aceptadas.append(t)
        else:
            # Tarea rechazada
            tareas_rechazadas.append({
                "tarea_nueva": tarea_nueva,
                "horario": horario_nuevo,
                "conflicto_con": conflicto["tarea"],
                "horario_conflicto": conflicto["horario"]
            })

    return tareas_aceptadas, tareas_rechazadas

# --------------------------
# Funciones GPIO MODIFICADAS
# --------------------------

def boton_rojo():
    """
    Graba el audio, lo transcribe, y procesa la tarea con Hugging Face.
    Guarda las tareas aceptadas y narra las rechazadas.
    """
    if not GPIO_INSTALADO:
        print("GPIO no está instalado. No se puede usar el botón rojo.")
        return
    if not client or not whisper_model:
        reproducir_texto("Error de inicialización de los modelos. Por favor, revisa la consola.")
        return

    archivo = grabar_mientras_pulsado_rojo()
    if not archivo:
        return

    texto_input = transcribir_audio(archivo)

    # 1. Recuperar datos
    tareas_actuales = cargar_acciones()
    historico = cargar_historial()

    # 2. Guardar input del usuario en el histórico
    historico.append({"input": texto_input, "timestamp": datetime.now().isoformat()})
    
    # 3. Enviar al modelo de Llama para extraer tareas
    messages = [
        {"role": "system", "content": CONTEXTO_TAREAS},
        {"role": "user", "content": texto_input}
    ]
    
    print("\nEnviando transcripción a Hugging Face...")
    try:
        completion = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=messages
        )
        respuesta_modelo = completion.choices[0].message.content.strip()
    except Exception as e:
        mensaje_error = f"Error de comunicación con Hugging Face: {e}. No se pudo procesar la tarea."
        print(mensaje_error)
        reproducir_texto(mensaje_error)
        save_json(ARCHIVO_HISTORIAL, historico)
        return
        
    print("\nRespuesta del Modelo (Tareas):")
    print(respuesta_modelo, "\n")

    if respuesta_modelo.upper() == "ERROR":
        mensaje = f"No te entendí. Dijiste: {texto_input}"
        print(f"{mensaje}")
        reproducir_texto(mensaje)
        save_json(ARCHIVO_HISTORIAL, historico)
        return

    try:
        nuevas_tareas = json.loads(respuesta_modelo)
        # Ajustar la estructura de la tarea para el resto del código
        nuevas_tareas = [{"tarea": t.get("tarea"), "horario": t.get("horario")} for t in nuevas_tareas if t.get("tarea") and t.get("horario")]
    except:
        mensaje = f"El sistema devolvió un formato incorrecto. Dijiste: {texto_input}"
        print(f"{mensaje}")
        reproducir_texto(mensaje)
        save_json(ARCHIVO_HISTORIAL, historico)
        return

    # 4. Mezclar tareas y verificar conflictos
    tareas_aceptadas, tareas_rechazadas = merge_tasks(tareas_actuales, nuevas_tareas, buffer_minutes=BUFFER_MINUTES)
    
    # 5. Gestionar Tareas Aceptadas
    if tareas_aceptadas:
        guardar_acciones(tareas_actuales) # Guardar las tareas aceptadas en acciones.json
        
        # Guardar en el historial
        historico.append({
            "resultado_aceptado": [{"nombre": t["tarea"], "hora": t["horario"], "fecha": datetime.now().strftime("%Y-%m-%d"), "realizado": False} for t in tareas_aceptadas],
            "timestamp": datetime.now().isoformat()
        })
        
        nombres_aceptados = [t['tarea'] for t in tareas_aceptadas]
        horarios_aceptados = [t['horario'] for t in tareas_aceptadas]
        
        if len(tareas_aceptadas) == 1:
             mensaje_confirmacion = f"Acción guardada: {nombres_aceptados[0]} a las {horarios_aceptados[0]}"
        else:
             mensaje_confirmacion = f"Se guardaron {len(tareas_aceptadas)} nuevas acciones. Revisa la próxima con el botón verde."
             
        print(f"{mensaje_confirmacion}")
        reproducir_texto(mensaje_confirmacion)
        
    # 6. Gestionar Tareas Rechazadas
    if tareas_rechazadas:
        texto_para_narracion = narrar_conflicto(tareas_rechazadas)
        print(f"\n🔊 Narración de conflicto: {texto_para_narracion}")
        reproducir_texto(texto_para_narracion)
        
        # Guardar el registro de rechazo y el texto narrado al histórico
        historico.append({
            "resultado_rechazado": tareas_rechazadas,
            "timestamp": datetime.now().isoformat(),
            "motivo": f"Conflicto de horario con margen de {BUFFER_MINUTES} minutos",
            "texto_narrado_conflicto": texto_para_narracion 
        })
        
    if not tareas_aceptadas and not tareas_rechazadas:
        print("ℹ️ No se encontraron tareas válidas en tu mensaje.")
        reproducir_texto("No se encontraron tareas válidas en tu mensaje.")

    # 7. Guardar el historial final
    save_json(ARCHIVO_HISTORIAL, historico)


def boton_verde():
    """Muestra la próxima acción o detiene la alerta (primer código)."""
    global alerta_activa, detener_alerta
    if not GPIO_INSTALADO:
        print("GPIO no está instalado. No se puede usar el botón verde.")
        return

    if alerta_activa:
        detener_alerta = True
        reproducir_texto("Alerta detenida")
        print("Alerta detenida")
    else:
        prox = obtener_proxima_accion()
        if prox:
            mensaje = f"Próxima acción: {prox[1]} a las {prox[0].strftime('%H:%M')}"
            reproducir_texto(mensaje)
            print(mensaje)
        else:
            mensaje = "No hay más acciones programadas hoy"
            reproducir_texto(mensaje)
            print(mensaje)

# --------------------------
# Monitoreo y Alerta Automática
# --------------------------
def monitor_botones():
    """Monitorea el estado de los botones (primer código)."""
    if not GPIO_INSTALADO:
        print("El monitoreo de botones está inactivo.")
        return

    rojo_ultimo = GPIO.input(PIN_ROJO)
    verde_ultimo = GPIO.input(PIN_VERDE)
    t_ultimo_verde = 0

    while True:
        ahora = time.time()

        rojo_actual = GPIO.input(PIN_ROJO)
        verde_actual = GPIO.input(PIN_VERDE)

        if rojo_ultimo == 1 and rojo_actual == 0:
            threading.Thread(target=boton_rojo, daemon=True).start()

        if verde_actual != verde_ultimo:
            if ahora - t_ultimo_verde > DEBOUNCE:
                t_ultimo_verde = ahora
                # Solo reaccionar al flanco de bajada (presionar) para el botón verde
                if verde_actual == 0:
                    threading.Thread(target=boton_verde, daemon=True).start()
            verde_ultimo = verde_actual
        
        rojo_ultimo = rojo_actual
        time.sleep(0.05)


def revisar_acciones():
    """Monitorea y lanza alertas automáticas (primer código)."""
    global alerta_activa, detener_alerta

    while True:
        if not alerta_activa:
            acciones = cargar_acciones()
            ahora = datetime.now()

            for acc in acciones:
                try:
                    # Usar 'horario' y 'tarea' del nuevo formato
                    hora_accion = datetime.strptime(acc["horario"], "%H:%M").replace(
                        year=ahora.year, month=ahora.month, day=ahora.day
                    )

                    fin_alerta = hora_accion + timedelta(minutes=BUFFER_MINUTES)

                    # 1. Estamos dentro de la ventana de 5 min
                    if hora_accion <= ahora <= fin_alerta:

                        alerta_activa = True
                        detener_alerta = False
                        inicio = time.time()

                        print(f"ALERTA ACTIVA: {acc['tarea']} a las {acc['horario']}")
                        
                        while not detener_alerta and (time.time() - inicio) < 300: # Alerta por 5 min máx
                            reproducir_texto(f"Recuerda: {acc['tarea']}")
                            time.sleep(30)

                        # Alerta terminada (por botón o por tiempo)
                        
                        # 2. actualizar historial
                        historial = cargar_historial()
                        for h in historial:
                            # Buscar la acción en el historial por nombre y hora para actualizar el estado
                            if h.get("nombre") == acc["tarea"] and h.get("hora") == acc["horario"]:
                                h["realizado"] = detener_alerta # True si se pulsó el botón
                        guardar_historial(historial)

                        # 3. borrar de acciones.json
                        acciones = [a for a in acciones if not (a["tarea"] == acc["tarea"] and a["horario"] == acc["horario"])]
                        guardar_acciones(acciones)

                        alerta_activa = False

                    # 4. Se pasó la ventana sin atender => realizado=False
                    elif ahora > fin_alerta:
                        
                        # actualizar historial
                        historial = cargar_historial()
                        for h in historial:
                            if h.get("nombre") == acc["tarea"] and h.get("hora") == acc["horario"]:
                                h["realizado"] = False
                        guardar_historial(historial)

                        # borrar de acciones.json
                        acciones = [a for a in acciones if not (a["tarea"] == acc["tarea"] and a["horario"] == acc["horario"])]
                        guardar_acciones(acciones)

                except Exception as e:
                    print(f"Error revisando acción {acc.get('tarea', 'desconocida')}: {e}")

        time.sleep(5)


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    if not GPIO_INSTALADO:
         print("### EJECUCIÓN SIN GPIO ###")
         print("El sistema no puede interactuar con los botones físicos.")
         print("Para probar la lógica, llama directamente a 'boton_rojo()' o 'boton_verde()'.")
         
    print("Sistema listo. Botón rojo para grabar acción, verde para ver próxima acción o detener alerta.")
    
    if GPIO_INSTALADO:
        threading.Thread(target=monitor_botones, daemon=True).start()
    
    threading.Thread(target=revisar_acciones, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Saliendo...")
        if GPIO_INSTALADO:
            GPIO.cleanup()