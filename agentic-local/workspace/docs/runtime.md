# Ejecucion local

## Memoria GPU

El servidor de chat debe usar `LLAMA_PARALLEL=1` en equipos con 4 GB de VRAM.

## Modelos auxiliares

El router se configura con `--models-max 1` para mantener un unico modelo cargado. OCR se ejecuta solamente durante trabajos de ingesta y se descarga al terminar.
