# Agentic Local

App local ampliable para usar un modelo GGUF servido por `llama.cpp` como agente con herramientas basicas de filesystem.
Por defecto el modelo se arranca fuera de Docker con `../llama.cpp/build-vulkan/bin/llama-server`, para que pueda aprovechar mejor la GPU/Vulkan del host.
El modelo por defecto es `../model/LFM2.5-1.2B-Thinking-Q4_K_M.gguf`.

## Arranque recomendado

```bash
cd agentic-local
./start-host-gpu.sh
```

Si Docker en tu usuario pide permisos, ejecuta una vez `sudo -v` antes del script o lanza el script desde una terminal interactiva para que pueda pedir la contraseña.

UI: <http://localhost:8000>

API del modelo: <http://localhost:8080>

Para parar:

```bash
./stop-host-gpu.sh
```

Logs del modelo: `agentic-local/logs/llama-server.log`

## Nota sobre el modelo

El fichero BF16 anterior devolvia texto incoherente incluso en pruebas directas contra `llama-server`.
Con `Q4_K_M` las respuestas son coherentes; al ser un modelo `Thinking`, la app usa `response_format` con JSON Schema para forzar llamadas de herramienta estructuradas.

## Alternativa todo en Docker

El servicio `llm` sigue disponible como perfil opcional, aunque en esta maquina Docker no estaba viendo la GPU:

```bash
cd agentic-local
sudo docker compose --profile container-llm up --build
```

El agente solo puede leer y escribir dentro de `agentic-local/workspace`, que se monta como `/workspace` dentro del contenedor.

## Herramientas incluidas

- `list_dir`: lista directorios.
- `glob`: busca rutas por patron glob.
- `find`: busca texto en archivos UTF-8.
- `read_file`: lee archivos.
- `write_file`: crea o reemplaza archivos.
- `edit_file`: reemplaza texto exacto en un archivo.
- `file_info`: devuelve metadatos basicos.

## Ampliar herramientas

1. Crear una funcion en `backend/tools/`.
2. Registrarla como `Tool(...)` en un `ToolRegistry`.
3. Importar ese registry en `backend/tools/__init__.py` o fusionarlo con el existente.

El bucle de agente esta en `backend/agent.py`. El modelo debe responder JSON:

```json
{"tool": "read_file", "args": {"path": "notes.md"}}
```

o:

```json
{"final": "Respuesta para el usuario"}
```
