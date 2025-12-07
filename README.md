# MAGDALENA: Asistente Cognitivo de Voz a Personas con Alzheimer Leve. 

Sistema basado en Raspberry Pi, Whisper y Llama para ayudar a personas mayores a gestionar sus tareas mediante voz.

---

## Descripción

Este proyecto permite registrar tareas por voz, almacenarlas, detectar conflictos horarios y generar alertas automáticas. Utiliza:

- **Whisper** para transcribir el audio.
- **Llama** de Hugging Face para interpretar órdenes.
- **Edge TTS** para la síntesis de voz.

---

## Características

<details>
<summary>Detalles de funcionamiento</summary>

- Registro de tareas mediante botón rojo.
- Transcripción automática del audio con faster-whisper.
- Interpretación de tareas y horarios por Llama en formato JSON.
- Gestión de conflictos horarios (±5 minutos).
- Alertas automáticas y notificaciones de tareas.
- Botón verde para mostrar la próxima tarea o detener alertas.

</details>

---

## Requisitos

<details>
<summary>Dependencias de software</summary>

- **Python 3.10+** y librerías:
```bash
pip install sounddevice numpy wave playsound edge-tts faster-whisper huggingface_hub

    Librería GPIO para Raspberry Pi:

sudo apt install python3-rpi.gpio

    Clave de Hugging Face:

HF_API_KEY = "TU_API_KEY"

</details> <details> <summary>Hardware</summary>

    Raspberry Pi con GPIO

    Micrófono USB

    Altavoz o salida de audio

    Botones conectados a:

        GPIO 17: botón rojo (grabar tareas)

        GPIO 27: botón verde (mostrar/detener alertas)

</details>
Estructura del proyecto

magdalena2.py
acciones.json
acciones_historial.json

Ejecución

python3 magdalena2.py
