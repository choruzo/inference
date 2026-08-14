# Agentic Local

Asistente local con chat GGUF, herramientas de filesystem y RAG sobre documentacion del workspace.
Por defecto el modelo se arranca fuera de Docker con `../llama.cpp/build-vulkan/bin/llama-server`, para que pueda aprovechar mejor la GPU/Vulkan del host.
El modelo por defecto es `../model/LFM2.5-1.2B-Thinking-Q4_K_M.gguf`.

## Arranque recomendado

```bash
cd agentic-local
./start-host-gpu.sh
```

Para RAG semantico, descarga y arranca el modelo de embeddings CPU antes de la app:

```bash
./download-rag-models.sh
./start-embeddings.sh
```

El worker OCR CPU se instala por separado con `../venv/bin/pip install -r backend/requirements-ocr.txt`. Para descargar tambien GOT-OCR2_0 como opcion avanzada: `DOWNLOAD_OCR_MODEL=1 ./download-rag-models.sh`.

El endpoint de embeddings usa `http://localhost:8091/v1` porque `8081` esta ocupado en este host. Puede cambiarse con `EMBEDDINGS_PORT` y `EMBEDDINGS_BASE_URL`.

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
- `rag_status`: muestra el estado durable del indice.
- `rag_reindex`: reindexa incrementalmente `workspace/docs`.
- `rag_search`: ejecuta retrieval hibrido local.

## RAG

Coloca Markdown, texto o codigo en `workspace/docs` y ejecuta:

```bash
../venv/bin/python -m backend.rag.cli reindex
EMBEDDINGS_BASE_URL=http://127.0.0.1:8091/v1 ../venv/bin/python -m backend.rag.cli embed
../venv/bin/python -m backend.rag.cli status
```

El historial del chat se conserva en localStorage hasta pulsar `Limpiar chat`, con un limite de 40 mensajes configurable mediante `MAX_CHAT_HISTORY_MESSAGES`. Chat y las respuestas RAG usan `MAX_RESPONSE_TOKENS=-1` por defecto: el servidor genera hasta que el modelo emite su fin de respuesta o agota la ventana de contexto. `RESPONSE_TIMEOUT=0` desactiva el timeout de estas generaciones. Chat no aplica un esquema JSON a la respuesta, para permitir que los modelos Thinking emitan `<think>...</think>`; las herramientas y el reranking siguen usando JSON Schema y un limite independiente mediante `MAX_STRUCTURED_TOKENS`.

El indice SQLite/FTS5 vive en `workspace/.rag_index`. Cada chunk conserva ruta, seccion, lineas, hashes y version del embedding. Las peticiones antiguas sin `modes` conservan el agente anterior; la UI envia un contrato explicito y Chat puro no ejecuta herramientas.

Conversion offline a Markdown:

```bash
../venv/bin/python -m backend.rag.cli convert documento.pdf
../venv/bin/python -m backend.rag.cli convert documento.docx
```

PDF digital usa `pypdf`; paginas sin texto e imagenes usan RapidOCR/ONNX en CPU con cache por hash. El Markdown se guarda en `workspace/.rag_sources`, se registra en `sources` y se indexa. Los pesos GOT-OCR2_0 se descargan como opcion avanzada bajo demanda, pero no quedan residentes en VRAM.

`OCR_PROVIDER=got_ocr` usa `stepfun-ai/GOT-OCR2_0` a traves de `backend/rag/got_ocr_worker.py`, un subproceso aislado (no se importa `torch`/`transformers` en el proceso principal de FastAPI). Requiere los pesos (`DOWNLOAD_OCR_MODEL=1 ./download-rag-models.sh`) y las dependencias pesadas (`../venv/bin/pip install -r backend/requirements-ocr-got.txt`, con `torch>=2.2` para compatibilidad con Python 3.12 en vez del `torch==2.0.1` que fija la model card). Sin ambas cosas falla con un error explicito indicando que instalar/descargar; con `MODEL_ROUTER_ENABLED=true`, y solo para este proveedor, `GotOcrVramSwap` descarga chat y embeddings antes de invocar el worker y los restaura al terminar (ver detalle mas abajo).

Se evaluo Docling para preservar layout complejo, tablas y JSON estructurado. No se incluye en el perfil inicial por su coste de dependencias y memoria en este host; el contrato de Markdown canonico y la tabla `sources` permiten incorporarlo despues como otro conversor sin cambiar el indice.

Evaluacion reproducible:

```bash
LLM_BASE_URL=http://127.0.0.1:8080/v1 EMBEDDINGS_BASE_URL=http://127.0.0.1:8091/v1 ../venv/bin/python -m backend.rag.cli evaluate
../venv/bin/pytest -q
```

La evaluacion genera respuestas RAG reales y calcula por separado retrieval, citas emitidas y grounding de la respuesta. El golden set esta en `workspace/evals/rag_golden.jsonl` y la ultima traza queda en `workspace/.rag_cache/last_evaluation.json`.

## Busqueda web

El modo Web usa SearXNG como proveedor principal y Tavily como fallback. Tavily solo se consulta si SearXNG falla, agota el timeout o entrega menos de `WEB_SEARCH_MIN_RESULTS` resultados utiles. Los resultados se normalizan y deduplican por URL; despues, las primeras paginas se descargan directamente y su contenido principal se extrae localmente. Cada salto de redireccion vuelve a validar el destino para bloquear protocolos no HTTP(S) y redes privadas, locales, loopback o link-local.

Arranca el servicio opcional de SearXNG antes de usar Web:

```bash
SEARXNG_SECRET="$(openssl rand -hex 32)" docker compose --profile web-search up -d searxng
```

La interfaz local de SearXNG queda en <http://localhost:8888>; la app lo consulta dentro de Docker en `http://searxng:8080`. El formato JSON esta habilitado en `searxng/settings.yml`.

Para habilitar el fallback, copia `.env.example` a `.env` y rellena `TAVILY_API_KEY`. No pongas el token en Compose, en el codigo ni en Git. La busqueda Tavily usa profundidad `basic`, no solicita respuesta generada ni contenido completo y, segun su documentacion vigente en agosto de 2026, cuesta 1 credito por consulta; el plan gratuito ofrece 1.000 creditos mensuales. Consulta la [API de busqueda](https://docs.tavily.com/documentation/api-reference/endpoint/search) y la [cuota vigente](https://docs.tavily.com/documentation/api-credits) antes de desplegar.

Variables disponibles:

- `WEB_SEARCH_PROVIDER`, `WEB_SEARCH_FALLBACK`, `WEB_SEARCH_MIN_RESULTS`, `WEB_SEARCH_LIMIT`
- `SEARXNG_URL`, `TAVILY_API_KEY`, `TAVILY_URL`
- `WEB_TIMEOUT`, `WEB_MAX_BYTES`, `WEB_MAX_REDIRECTS`
- `WEB_FETCH_RESULTS`, `WEB_FETCH_MAX_CHARS`, `WEB_RETRY_ATTEMPTS`, `WEB_RETRY_BACKOFF`, `WEB_USER_AGENT`

Para usar solo un proveedor, deja `WEB_SEARCH_FALLBACK` vacio. `WEB_SEARCH_PROVIDER=tavily` selecciona Tavily como principal. Chat con Web desactivado no ejecuta ninguna peticion de busqueda ni descarga de paginas.

## Configuracion RAG

- `RAG_ENABLED`, `RAG_DOCS_DIR`, `RAG_INDEX_DIR`, `RAG_SOURCES_DIR`, `RAG_CACHE_DIR`
- `RAG_CHUNK_TOKENS`, `RAG_CHUNK_OVERLAP`, `RAG_TOP_K`, `RAG_RERANK_TOP_K`, `RAG_CONTEXT_TOKENS`, `RAG_MIN_VECTOR_SCORE`
- `EMBEDDINGS_PROVIDER`, `EMBEDDINGS_MODEL`, `EMBEDDINGS_BASE_URL`, `EMBEDDINGS_DIMENSIONS`, `EMBEDDINGS_BATCH_SIZE`
- `OCR_ENABLED`, `OCR_PROVIDER`, `OCR_MODEL`, `OCR_MODEL_DIR`
- `MODEL_ROUTER_ENABLED`, `MODEL_ROUTER_BASE_URL`, `MODEL_ROUTER_MAX_MODELS`, `MODEL_ROUTER_RESTORE_CHAT`
- `MAX_RESPONSE_TOKENS`, `MAX_STRUCTURED_TOKENS`, `MAX_CHAT_HISTORY_MESSAGES`, `RESPONSE_TIMEOUT`

Para probar la carga secuencial real, detiene primero cualquier arranque normal y ejecuta `MODEL_ROUTER_ENABLED=true ./start-host-gpu.sh`. El preset `rag-models.ini` usa los alias `local-gguf` y `bge-small-en-v1.5`, ambos con `load-on-startup = true` y `--models-max 2`: chat y embeddings conviven cargados a la vez (caben juntos en 4 GB, ver perfil de VRAM en `RAG_IMPLEMENTATION_PLAN.md`), igual que en el modo normal sin router.

En una GTX 1050 de 4 GB, el chat mantiene `LLAMA_CTX_SIZE=128000` y `LLAMA_PARALLEL=1`. Embeddings se sirve en CPU por defecto (o convive en GPU bajo el router). Ni RapidOCR/tesseract (CPU) ni la conversion de PDF/DOCX digital tocan el router: chat y embeddings se quedan cargados durante toda la ingesta normal. Solo `OCR_PROVIDER=got_ocr` (GOT-OCR2_0, un VLM que si ocupa VRAM real) dispara `GotOcrVramSwap`: descarga chat y embeddings mediante `/models/unload` justo antes de invocar el worker, y los restaura con `/models/load` al terminar, incluso si el worker falla.

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
