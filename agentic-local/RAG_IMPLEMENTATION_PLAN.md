# Plan de implementacion RAG para Agentic Local

## Objetivo

Convertir el agente actual en un asistente local con tres modos de trabajo:

- **Chat**: responde solo con el modelo local, sin herramientas externas.
- **RAG**: consulta un indice de documentacion local con chunking, embeddings, busqueda hibrida, reranking y citas.
- **Web**: usa busqueda en internet mediante `web_search` y lectura de paginas mediante `web_fetch`.

Los modos deben poder activarse o desactivarse desde el cuadro de chat usando un boton con simbolo `+`, que abre controles compactos. La interfaz tambien debe mostrar una ventana colapsable con tokens o texto de razonamiento/traza cuando el backend lo exponga.

## Principios RAG

- No meter documentos completos en contexto salvo casos pequenos y controlados.
- Recuperar pocos fragmentos buenos, no muchos fragmentos mediocres.
- Separar recuperacion, reranking, generacion y citacion.
- Citar siempre la fuente concreta usada: archivo, titulo, seccion y rango aproximado de lineas o chunk.
- Mantener el texto citado verificable contra el contenido original.
- No mezclar informacion recuperada con suposiciones del modelo sin marcarlo.
- Guardar metadatos suficientes para auditoria: ruta, hash, timestamp, chunk id, offsets y estrategia de chunking.
- Permitir reindexacion incremental por cambios de archivo.

## Arquitectura Propuesta

### Componentes

- `backend/rag/ingest.py`: descubre documentos, extrae texto, divide en chunks y actualiza el indice.
- `backend/rag/chunking.py`: estrategias de chunking por tipo de documento.
- `backend/rag/embeddings.py`: cliente de embeddings local o remoto configurable.
- `backend/rag/store.py`: almacenamiento vectorial y metadatos.
- `backend/rag/search.py`: busqueda hibrida texto + vector.
- `backend/rag/rerank.py`: reranking de candidatos antes de generar respuesta.
- `backend/rag/citations.py`: construccion y validacion de citas.
- `backend/tools/web.py`: herramientas `web_search` y `web_fetch`.
- `backend/modes.py`: validacion de modos activos por conversacion.
- `frontend/app.js`: selector `+`, estados de modo, panel de razonamiento colapsable y renderizado de citas.

### Flujo RAG

1. Usuario envia pregunta con modo RAG activo.
2. Backend normaliza la pregunta y detecta filtros opcionales.
3. `search.hybrid_search()` obtiene candidatos:
   - vector search por similitud semantica;
   - keyword/BM25 o FTS para terminos exactos;
   - fusion por Reciprocal Rank Fusion.
4. `rerank.rerank()` reordena los mejores candidatos.
5. Se construye un contexto compacto con los chunks top-k.
6. El modelo responde usando solo el contexto recuperado cuando la pregunta sea factual.
7. `citations.py` adjunta citas por afirmacion o por parrafo.
8. La UI muestra respuesta, fuentes y traza/razonamiento colapsable.

## Ingestion y Chunking

### Formatos iniciales

- Markdown: `.md`, `.mdx`
- Texto: `.txt`
- Codigo/documentacion: `.py`, `.js`, `.ts`, `.tsx`, `.html`, `.css`, `.yml`, `.yaml`, `.json`
- PDF: fase posterior, solo si se anade extractor fiable.

### Estrategia

- Markdown:
  - dividir por encabezados;
  - conservar jerarquia de titulos como metadatos;
  - chunk objetivo: 400-900 tokens;
  - overlap: 80-120 tokens.
- Codigo:
  - dividir por funciones/clases cuando sea posible;
  - fallback por bloques de lineas;
  - guardar simbolos detectados.
- Texto plano:
  - dividir por parrafos;
  - combinar parrafos cortos hasta objetivo.

### Metadatos por chunk

```json
{
  "chunk_id": "sha256(path + offset + content_hash)",
  "doc_id": "sha256(path)",
  "path": "docs/example.md",
  "title": "Titulo detectado",
  "section": "Seccion > Subseccion",
  "start_line": 10,
  "end_line": 42,
  "content_hash": "sha256(chunk_text)",
  "document_hash": "sha256(full_text)",
  "token_count": 612,
  "created_at": 1780000000,
  "updated_at": 1780000000
}
```

## Embeddings

### Requisitos

- Embeddings rapidos, locales si es posible.
- Dimensiones estables y versionadas.
- Reindexar automaticamente si cambia el modelo de embeddings.
- Deben caber junto al chat actual o cargarse secuencialmente sin desbordar VRAM.

### Perfil de VRAM medido

Con `LFM2.5-1.2B-Thinking-Q4_K_M.gguf`, `LLAMA_CTX_SIZE=128000` y `LLAMA_PARALLEL=1`, el chat usa aproximadamente 2.38 GiB de una GTX 1050 de 4 GB. Queda un margen teorico de 1.6-1.7 GiB, pero se debe reservar al menos 500-700 MiB para picos, buffers, escritorio y fragmentacion. Por tanto:

- embeddings pequenos pueden convivir con el chat;
- OCR debe ejecutarse preferentemente por cola y con carga secuencial;
- modelos auxiliares grandes no deben quedar cargados permanentemente.

### Opciones

- Fase 1: usar un modelo de embeddings local servido por un proceso separado o una libreria Python.
- Fase 2: permitir backend configurable:
  - local embeddings;
  - endpoint OpenAI-compatible;
  - otro servidor HTTP local.

### Modelo recomendado

Usar `bge-small-en-v1.5` en GGUF cuantizado como primera opcion para embeddings:

- Repo base: `BAAI/bge-small-en-v1.5`
- GGUF recomendado: `smarttasks/bge-small-en-v1.5-GGUF`
- Variante: `bge-small-en-v1.5-Q8_0.gguf` si se prioriza fidelidad.
- Variante minima: `bge-small-en-v1.5-Q4_K_M.gguf` si se prioriza memoria.
- Tamano aproximado GGUF: 29-37 MB.
- Dimension: 384.
- Contexto del embedding: normalmente 512 tokens; por eso el chunk objetivo no debe superar 400-500 tokens para este modelo si se quiere maxima fidelidad.

Motivo: es suficientemente pequeno para caber junto al chat, tiene formato GGUF, es compatible con servidores estilo llama.cpp/ggml y es una base RAG conocida. `nomic-embed-text-v1.5` tambien es buena opcion, pero es mas grande; encaja mejor si se sirve en CPU, ONNX int8/q4 o como modelo cargado bajo demanda.

Referencias:

- `https://huggingface.co/BAAI/bge-small-en-v1.5`
- `https://huggingface.co/smarttasks/bge-small-en-v1.5-GGUF`
- `https://huggingface.co/nomic-ai/nomic-embed-text-v1.5`
- `https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF`

### Configuracion

Variables sugeridas:

```bash
EMBEDDINGS_PROVIDER=local
EMBEDDINGS_MODEL=bge-small-en-v1.5
EMBEDDINGS_BASE_URL=http://localhost:8081/v1
RAG_INDEX_DIR=/workspace/.rag_index
RAG_CHUNK_TOKENS=450
RAG_CHUNK_OVERLAP=80
RAG_TOP_K=8
RAG_RERANK_TOP_K=4
```

Si se usa `nomic-embed-text-v1.5`, subir `RAG_CHUNK_TOKENS` solo tras validar el contexto real servido. Nomic requiere prefijos de tarea, por ejemplo `search_query:` para preguntas y `search_document:` para documentos.

## Almacenamiento e Indice

### Recomendacion inicial

Usar SQLite como base durable:

- tabla `documents`;
- tabla `chunks`;
- tabla `embeddings`;
- FTS5 para busqueda textual;
- vector store simple local si se usa extension disponible, o almacenamiento NumPy/FAISS si se acepta dependencia.

### Alternativas

- SQLite + FTS5 + embeddings en archivos `.npy`: simple y portable.
- Chroma/FAISS: mejor vector search, mas dependencias.
- LanceDB: buena opcion si se quiere evolucionar a datasets grandes.

Para este proyecto, empezar con SQLite + FTS5 + almacenamiento vectorial simple es suficiente. Si el corpus crece, migrar a FAISS o LanceDB.


### Esquema minimo

Tablas:

- `documents`: una fila por archivo fuente normalizado.
- `chunks`: texto indexable, metadatos, lineas, hash y estado.
- `chunk_fts`: tabla virtual FTS5 sobre `chunks.content`.
- `embeddings`: vector binario o ruta a vector `.npy`, dimension, modelo y hash.
- `sources`: fuente original para documentos convertidos, por ejemplo PDF o DOCX.

Campos importantes:

```sql
documents(id, path, source_type, normalized_path, title, content_hash, indexed_at)
chunks(id, document_id, content, start_line, end_line, section, token_count, content_hash)
embeddings(chunk_id, model, dimensions, vector_blob, vector_path, created_at)
sources(id, original_path, normalized_markdown_path, converter, converter_version, ocr_model, created_at)
```

Para corpora pequenos, guardar `vector_blob` en SQLite es aceptable. Para corpora medianos, guardar vectores en `.npy` o `.faiss` y mantener SQLite como catalogo.

## PDF, Word y OCR a Markdown

### Regla de ingestion

Todo documento no textual debe convertirse primero a Markdown canonico y luego indexarse como Markdown. El indice nunca debe depender directamente del binario original.

Flujo:

1. Detectar tipo: PDF digital, PDF escaneado, DOCX, imagen.
2. Extraer texto estructurado si existe capa de texto.
3. Si falta texto o la calidad es baja, enviar paginas a OCR.
4. Generar `.md` normalizado en `/workspace/.rag_sources/`.
5. Guardar relacion entre original y Markdown en `sources`.
6. Indexar el Markdown resultante.

### PDF digital

- Primero intentar extraccion directa con una libreria de PDF.
- Conservar encabezados, tablas simples y paginas.
- Si el texto extraido tiene demasiados caracteres corruptos o paginas vacias, pasar esas paginas por OCR.

### DOCX/Word

- Convertir a Markdown con un conversor determinista.
- Mantener titulos, listas, tablas e imagenes referenciadas.
- Si contiene imagenes escaneadas, extraerlas y pasarlas por OCR.

### OCR recomendado

Opciones desde Hugging Face:

- `stepfun-ai/GOT-OCR2_0`: OCR general, multilingual, 0.7B, pesa alrededor de 1.44 GB en safetensors BF16. Es candidato razonable para ejecucion secuencial.
- `nanonets/Nanonets-OCR-s`: orientado a image-to-markdown y pdf2markdown, muy bueno para estructura, tablas y documentos complejos, pero basado en Qwen2.5-VL-3B y con pesos mucho mayores. No debe convivir con el chat en una GTX 1050 de 4 GB; usar solo bajo demanda, CPU/offload o maquina externa.

Referencias:

- `https://huggingface.co/stepfun-ai/GOT-OCR2_0`
- `https://huggingface.co/nanonets/Nanonets-OCR-s`

### Politica de OCR

- No cargar OCR durante una conversacion normal.
- Ejecutar OCR solo en jobs de ingestion.
- Antes de iniciar OCR, el orquestador debe descargar o parar el modelo principal y el modelo de embeddings.
- Mientras OCR esta activo, debe quedar cargado solo el modelo OCR.
- Al terminar OCR, descargar el modelo OCR y liberar VRAM.
- Despues de generar Markdown, reactivar embeddings para indexar el texto normalizado.
- Procesar por paginas con cola y cache.
- Guardar imagen de pagina, Markdown producido y hash.
- Permitir corregir manualmente el Markdown antes de indexar.
- Registrar modelo OCR, version y parametros.

## Router llama.cpp y Carga Secuencial

Para evitar desbordar VRAM, usar router de `llama-server` para cargar modelos bajo demanda:

```bash
llama-server \
  --models-dir ./models \
  --models-max 1 \
  --no-models-autoload \
  --host 127.0.0.1 \
  --port 8080
```

Regla:

- `--models-max 1` fuerza que solo haya un modelo cargado cuando se use el router compartido.
- Durante chat normal, cargar solo el modelo principal.
- Durante generacion de embeddings, descargar el chat si se necesita garantizar VRAM libre, cargar solo embeddings, procesar batch y descargar embeddings.
- Durante OCR de ingesta, descargar siempre chat y embeddings; cargar solo el modelo OCR; convertir PDF/Word/imagenes a Markdown; descargar OCR al terminar.
- OCR nunca debe ejecutarse durante una conversacion normal ni quedar como modelo residente.
- Reranker grande sigue la misma regla que OCR: carga bajo demanda, job corto, descarga.

Alternativa con presets:

```ini
[chat]
model = /models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf
ctx-size = 128000
parallel = 1

[embeddings]
model = /models/bge-small-en-v1.5-Q8_0.gguf
embedding = true
pooling = cls
ctx-size = 512

[reranker]
model = /models/bge-reranker-v2-m3.gguf
embedding = true
pooling = rank
```

Nota: validar cada preset contra la build local de llama.cpp, porque no todos los modelos de OCR/VLM estaran disponibles en GGUF o soportados por el router. Si el OCR requiere Transformers, el orquestador debe parar/descargar los modelos de llama.cpp antes de arrancar el worker Python de OCR, y volver a levantar chat/embeddings al terminar.

Referencias:

- `https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md`
- `https://huggingface.co/blog/ggml-org/model-management-in-llamacpp`

## Busqueda Hibrida

### Pipeline

1. `keyword_search(query)` con FTS5.
2. `vector_search(query_embedding)`.
3. Fusionar resultados con RRF:

```text
score = sum(1 / (k + rank_i))
```

4. Penalizar chunks demasiado cortos, duplicados o del mismo bloque.
5. Enviar top `RAG_TOP_K` al reranker.

### Buenas practicas

- Mantener diversidad de documentos.
- No devolver 8 chunks casi iguales.
- Priorizar coincidencias exactas para nombres de funciones, rutas, comandos y errores.
- Guardar en la traza la razon de seleccion de cada chunk.

## Reranking

### Fase 1

Reranking local con el propio LLM:

- entrada: pregunta + candidatos compactos;
- salida JSON con ids ordenados y motivo breve;
- limite de candidatos: 8-12.

### Fase 2

Anadir reranker dedicado si hay modelo local disponible:

- cross-encoder;
- endpoint HTTP;
- modelo pequeno especializado.

### Salida esperada

```json
{
  "ranked_chunk_ids": ["chunk_a", "chunk_b"],
  "reasons": {
    "chunk_a": "Define directamente el parametro solicitado"
  }
}
```

## Generacion con Citas

El prompt de respuesta debe exigir:

- responder en el idioma del usuario;
- citar cada afirmacion factual relevante;
- decir "no lo encuentro en la documentacion indexada" cuando falte evidencia;
- separar inferencias de hechos citados;
- evitar inventar rutas, APIs o comandos.

Formato interno sugerido:

```json
{
  "answer": "Texto final con citas [1].",
  "citations": [
    {
      "id": 1,
      "path": "docs/install.md",
      "title": "Instalacion",
      "section": "GPU",
      "start_line": 20,
      "end_line": 35,
      "quote": "fragmento corto opcional"
    }
  ]
}
```

La UI debe renderizar las citas al final y permitir saltar a la fuente cuando sea un archivo del workspace.

## Herramientas Web

### `web_search`

Responsabilidad:

- buscar paginas relevantes;
- devolver titulo, url, snippet, fecha si existe y ranking.

Schema:

```json
{
  "query": "texto",
  "limit": 5,
  "recency_days": null
}
```

### `web_fetch`

Responsabilidad:

- descargar una URL;
- extraer contenido principal;
- devolver texto limpio, titulo, url, fecha y metadatos.

Schema:

```json
{
  "url": "https://example.com/page",
  "max_chars": 20000
}
```

### Seguridad

- timeouts estrictos;
- limite de tamano;
- bloquear protocolos no HTTP/HTTPS;
- no ejecutar scripts;
- registrar URL final tras redirects;
- mostrar claramente que la fuente viene de internet.

## Modos del Chat

### Estados

```json
{
  "chat": true,
  "rag": false,
  "web": false,
  "reasoning_panel": true
}
```

### Reglas

- Chat solo: no usar RAG ni web.
- RAG activo: buscar primero en indice local.
- Web activo: puede usar `web_search` y `web_fetch`.
- RAG + Web activo: preferir documentacion local; usar web para informacion externa o actual.
- Si ningun modo especializado esta activo, comportamiento actual de chat.

### UX

- En el cuadro de chat, boton `+`.
- Al pulsar `+`, abrir un pequeno menu con toggles:
  - RAG
  - Web
  - Chat
  - Razonamiento
- Mostrar chips compactos de modos activos junto al input.
- Persistir preferencia en `localStorage`.
- Enviar modos activos en `/api/chat`.

## Panel de Razonamiento y Traza

### Objetivo

Mostrar informacion util sin saturar la respuesta principal.

### Contenido

- razonamiento si el backend/modelo lo expone;
- tool calls;
- chunks recuperados;
- scores de busqueda;
- resultados de reranking;
- fuentes consultadas;
- estado final: `final`, `max_steps`, `llm_timeout`, etc.

### UX

- Panel colapsable por mensaje.
- Estado inicial colapsado.
- Etiquetas por tipo:
  - `tool`
  - `rag`
  - `web`
  - `rerank`
  - `model`
- Boton para copiar traza JSON.

Nota: si el modelo devuelve `<think>...</think>`, el backend debe extraerlo y pasarlo como campo separado, no mezclarlo con la respuesta final.

## Cambios de API

### Request

```json
{
  "message": "pregunta",
  "history": [],
  "modes": {
    "chat": true,
    "rag": true,
    "web": false,
    "reasoning_panel": true
  }
}
```

### Response

```json
{
  "answer": "respuesta",
  "citations": [],
  "reasoning": "texto opcional",
  "trace": [],
  "retrieval": {
    "query": "pregunta",
    "chunks": [],
    "reranked": []
  },
  "workspace": "/workspace",
  "stopped": "final"
}
```

## Conclusiones de Proyectos RAG Revisados

Las referencias revisadas sugieren que el sistema debe crecer por capas, no por contexto bruto:

- Haystack estructura RAG como pipelines de componentes: indexing, retrievers, routers, rerankers y generadores. Esta arquitectura encaja con este proyecto porque permite activar Chat, RAG, Web, OCR e ingesta como flujos separados.
- GraphRAG separa busquedas por alcance: busqueda local para preguntas concretas, global para preguntas sobre todo el corpus y DRIFT para ampliar desde entidades a comunidades. Para este proyecto conviene empezar con RAG local/hibrido y dejar una fase posterior para resumenes jerarquicos o grafo ligero.
- LightRAG refuerza cuatro ideas utiles: chunking configurable, roles separados para extraction/query/keywords/VLM, evaluacion con contextos recuperados y escalabilidad eliminando cuellos de botella. Para esta maquina, eso se traduce en colas de ingesta, batches de embeddings y trazas de retrieval.
- RAGFlow y Docling tratan documentos complejos como un problema de document understanding antes del RAG. La conclusion practica es convertir PDF/Word/imagenes a Markdown estructurado en una fase offline, revisar/cachear ese Markdown y solo despues indexarlo.
- Docling aporta una ruta concreta para crecer sin perder estructura: exportar Markdown/JSON, conservar tablas, paginas y jerarquia, y usar chunking estructural/hibrido en vez de cortes arbitrarios.

Referencias:

- `https://docs.haystack.deepset.ai/docs/pipelines`
- `https://docs.haystack.deepset.ai/docs/retrievers`
- `https://microsoft.github.io/graphrag/index/overview/`
- `https://microsoft.github.io/graphrag/query/overview/`
- `https://github.com/HKUDS/LightRAG`
- `https://ragflow.net/docs`
- `https://docling-project.github.io/docling/reference/cli/`
- `https://docling.org/`

## Fases de Implementacion Separadas

### Fase 0: Preparacion y Contratos

Objetivo: dejar claros los limites del sistema antes de implementar RAG.

Tareas:

- Definir contratos JSON para `modes`, `retrieval`, `citations`, `reasoning` y `trace`.
- Definir estructura de carpetas: `/workspace/docs`, `/workspace/.rag_index`, `/workspace/.rag_sources`, `/workspace/.rag_cache`.
- Definir variables de entorno para RAG, embeddings, router, OCR y Web.
- Anadir feature flags para activar cada capacidad sin romper el chat actual.
- Documentar limites de VRAM: chat 128k, embeddings pequenos, OCR solo ingesta.

Criterio de salida:

- El chat actual sigue funcionando igual con `modes` ausente.
- Existe schema validado para requests/responses nuevas.

### Fase 1: UI de Modos y Panel de Traza

Objetivo: que el usuario pueda elegir Chat, RAG, Web y razonamiento desde el composer.

Tareas:

- Anadir boton `+` en el cuadro de chat.
- Mostrar menu compacto con toggles: Chat, RAG, Web, Razonamiento.
- Persistir modos en `localStorage`.
- Enviar `modes` en `/api/chat`.
- Mostrar chips de modos activos.
- Crear panel colapsable por mensaje para `reasoning`, `trace`, tool calls, retrieval y fuentes.

Criterio de salida:

- Chat solo no llama herramientas.
- RAG/Web desactivados no aparecen en trace.
- El panel colapsable puede copiar trace JSON.

### Fase 2: Ingesta Textual Canonica

Objetivo: indexar Markdown/texto/codigo sin embeddings todavia.

Tareas:

- Implementar `backend/rag/ingest.py`.
- Soportar `.md`, `.txt`, `.py`, `.js`, `.ts`, `.tsx`, `.html`, `.css`, `.json`, `.yaml`, `.yml`.
- Normalizar contenido a UTF-8.
- Calcular `document_hash` y `content_hash`.
- Guardar documento y chunks con lineas, seccion, ruta y token count.
- Implementar reindexacion incremental: solo recalcular documentos cuyo hash cambio.

Estrategia para calidad:

- Markdown: dividir por jerarquia de encabezados y luego ajustar por tokens.
- Codigo: preferir clases/funciones; fallback por bloques de lineas.
- Texto plano: parrafos con overlap moderado.
- Nunca mezclar secciones inconexas en un chunk.

Criterio de salida:

- Reindexar dos veces sin cambios no duplica chunks.
- Cada chunk permite abrir la fuente original con lineas aproximadas.

### Fase 3: Base de Datos e Indice Textual

Objetivo: tener una base durable y auditable antes de anadir vectores.

Tareas:

- Crear SQLite en `/workspace/.rag_index/index.sqlite`.
- Crear tablas `documents`, `chunks`, `sources`, `ingest_jobs`.
- Crear tabla virtual FTS5 `chunk_fts`.
- Implementar migraciones versionadas.
- Implementar `rag_status` y `rag_reindex` como endpoints/herramientas internas.

Estrategia para eficiencia:

- Usar transacciones por batch.
- Mantener indices por `document_hash`, `path` y `updated_at`.
- Evitar cargar todo el corpus en memoria.

Criterio de salida:

- Busqueda textual encuentra terminos exactos, rutas, comandos y errores.
- La base puede borrarse y reconstruirse desde `/workspace/docs`.

### Fase 4: Retrieval Textual con Citas

Objetivo: primer RAG util sin embeddings.

Tareas:

- Implementar `keyword_search()` con FTS5.
- Crear `rag_search` que devuelva chunks, scores y metadatos.
- Construir prompt de generacion con contexto top-k.
- Exigir citas por parrafo o afirmacion factual.
- Responder claramente si no hay evidencia suficiente.

Estrategia para no perder calidad:

- Para queries con nombres exactos, priorizar FTS sobre semantica.
- Deduplicar chunks del mismo documento/seccion.
- Limitar contexto final a chunks diversos.

Criterio de salida:

- Preguntas literales devuelven respuestas con citas verificables.
- Preguntas sin evidencia no inventan.

### Fase 5: Embeddings Locales

Objetivo: anadir recuperacion semantica sin comprometer VRAM.

Tareas:

- Descargar/configurar `bge-small-en-v1.5` GGUF.
- Servir embeddings con llama.cpp u otro servidor OpenAI-compatible en `localhost:8081`.
- Implementar `backend/rag/embeddings.py`.
- Guardar vectores con modelo, dimension y version.
- Recalcular vectores solo para chunks nuevos o modificados.

Estrategia para escalar:

- Procesar embeddings por batches pequenos.
- Usar cola de jobs para no bloquear el chat.
- Si se usa router compartido, descargar chat antes de batches grandes.
- Mantener `RAG_CHUNK_TOKENS=450` para respetar el contexto corto de `bge-small`.

Criterio de salida:

- Vector search encuentra respuestas aunque la query no comparta palabras exactas.
- Cambiar `EMBEDDINGS_MODEL` invalida y recalcula vectores.

### Fase 6: Busqueda Hibrida y Fusion

Objetivo: combinar precision lexical y semantica sin degradar resultados al crecer.

Tareas:

- Implementar `vector_search()`.
- Implementar fusion RRF entre FTS y vector search.
- Anadir filtros por ruta, tipo, fecha, etiqueta y fuente.
- Anadir diversidad por documento/seccion.
- Registrar en trace rankings separados y ranking fusionado.

Estrategia para aumentar tamano manteniendo calidad:

- `candidate_pool`: traer 30-80 candidatos baratos.
- `diversity_pass`: reducir duplicados por documento.
- `context_budget`: seleccionar chunks hasta un presupuesto de tokens, no por numero fijo.
- `query_router`: detectar pregunta factual, conceptual, ruta exacta, comparacion o resumen global.
- `adaptive_top_k`: subir candidatos cuando la confianza es baja, bajarlos cuando hay match exacto.

Criterio de salida:

- En corpus pequeno y mediano, la respuesta mantiene citas correctas.
- El trace explica por que cada chunk entro al contexto.

### Fase 7: Reranking y Control de Contexto

Objetivo: mejorar precision antes de enviar contexto al modelo.

Tareas:

- Implementar reranking por LLM con salida JSON estricta.
- Preparar interfaz para reranker dedicado posterior.
- Limitar candidatos al reranker para controlar coste.
- Reordenar y recortar contexto final a top 3-8 chunks segun presupuesto.

Estrategia para eficiencia:

- Rerank solo si hay mas de N candidatos o baja confianza.
- Cachear rerank por `query_hash + candidate_ids`.
- No usar reranker grande residente; cargarlo bajo demanda si existe.

Criterio de salida:

- Reranking mejora o mantiene precision en golden queries.
- No aumenta latencia excesiva para queries simples.

### Fase 8: Evaluacion, Tracing y Golden Set

Objetivo: medir calidad antes de ampliar corpus.

Tareas:

- Crear `workspace/evals/rag_golden.jsonl`.
- Guardar preguntas, respuesta esperada, fuentes esperadas y tipo de query.
- Medir hit@k, citation accuracy, answer groundedness y latencia.
- Guardar traces de retrieval para inspeccion.
- Anadir smoke tests automatizados.

Estrategia de calidad:

- No aceptar cambios de chunking/embedding si empeoran golden queries.
- Medir precision de contexto, no solo si la respuesta suena bien.
- Revisar manualmente fallos por tipo: chunking, retrieval, rerank, prompt o cita.

Criterio de salida:

- Existe baseline reproducible antes de indexar documentos grandes.

### Fase 9: Documentos Complejos y OCR Offline

Objetivo: PDF/Word/imagenes entran por Markdown normalizado, no directamente al indice.

Tareas:

- Integrar conversor determinista para DOCX/PDF digital.
- Evaluar Docling como opcion principal para conversion a Markdown/JSON y chunking estructural.
- Implementar detector de PDF escaneado o texto de baja calidad.
- Implementar worker OCR solo para ingesta.
- Orquestar router: descargar chat/embeddings, cargar OCR, convertir, descargar OCR, reactivar embeddings.
- Guardar Markdown generado en `/workspace/.rag_sources/`.

Estrategia de calidad:

- OCR por pagina con cache por hash de imagen.
- Mantener pagina, bounding boxes si existen y tabla/imagen como metadatos.
- Permitir revision manual del Markdown antes de indexar documentos criticos.
- No indexar el binario original; indexar el Markdown canonico.

Criterio de salida:

- Un PDF digital y un PDF escaneado terminan como Markdown citables.
- OCR no queda cargado despues del job.

### Fase 10: Web Search y Web Fetch

Objetivo: busqueda externa opt-in, separada de RAG local.

Tareas:

- Implementar `web_search` con limite, timeout y metadatos.
- Implementar `web_fetch` con extraccion de contenido principal.
- Separar fuentes locales y web en la respuesta.
- Anadir politicas de seguridad: HTTP/HTTPS, tamano maximo, redirects, user agent y bloqueo de scripts.

Criterio de salida:

- Web desactivado no hace red.
- Web activado muestra URLs y fecha/metadatos cuando existan.

### Fase 11: Modos de Profesor

Objetivo: que el agente explique, no solo recupere.

Tareas:

- Anadir perfiles de respuesta: explicar, resumir, comparar, tutorial, diagnostico, quiz.
- Permitir comandos ligeros: `explica mas simple`, `dame ejemplo`, `hazme preguntas`, `muestra fuentes`.
- En modo profesor, usar RAG para grounding y el modelo para pedagogia.
- Mostrar citas sin romper la fluidez de la explicacion.

Criterio de salida:

- Las explicaciones largas siguen estando fundamentadas en citas.
- El usuario puede pedir profundizacion sin reconsultar todo el corpus si el trace/contexto sigue valido.

### Fase 12: Escalado del RAG

Objetivo: crecer de carpeta pequena a corpus grande sin perder calidad ni velocidad.

Tareas:

- Implementar indexado incremental por lotes.
- Anadir compactacion y limpieza de chunks huerfanos.
- Implementar resumenes por documento y seccion para preguntas globales.
- Evaluar grafo ligero de entidades solo si las preguntas globales fallan con RAG hibrido.
- Separar storage caliente: SQLite/FTS para metadatos y FAISS/LanceDB si vectores crecen demasiado.

Estrategia inspirada en GraphRAG/LightRAG:

- Local search para preguntas especificas.
- Summary search para preguntas sobre documentos completos.
- Global search posterior mediante resumenes jerarquicos, no metiendo todos los chunks en contexto.
- Grafo o entidades solo cuando haya valor claro: relaciones, dependencias, conceptos repetidos.

Criterio de salida:

- Aumentar el corpus no degrada golden queries.
- Las queries globales usan resumenes jerarquicos o grafo, no top-k ingenuo.

## Tests

### Unitarios

- chunking conserva lineas y metadatos;
- ingestion detecta cambios por hash;
- FTS encuentra terminos exactos;
- vector search devuelve candidatos esperados;
- RRF fusiona rankings de forma estable;
- citas apuntan a chunks existentes;
- `web_fetch` bloquea protocolos no permitidos.

### Integracion

- pregunta con RAG activo devuelve cita local;
- pregunta sin evidencia responde que no hay informacion suficiente;
- RAG + Web conserva fuentes separadas;
- Chat solo no llama herramientas;
- panel de razonamiento recibe `trace`;
- menu `+` persiste modos.

### Smoke test

1. Indexar `workspace/docs`.
2. Preguntar algo que aparece literalmente.
3. Verificar respuesta con cita.
4. Preguntar algo semantico.
5. Verificar recuperacion vectorial.
6. Activar Web y buscar informacion externa.
7. Verificar que se muestran URLs.

## Riesgos

- El modelo pequeno puede fallar estructurando JSON complejo.
- Reranking con el mismo modelo puede ser irregular.
- Internet requiere politicas de seguridad y timeouts.
- Contextos enormes pueden inducir respuestas lentas o menos enfocadas.
- Mostrar razonamiento bruto puede mezclar texto inutil con trazas utiles; conviene separar `reasoning` de `trace`.

## Decisiones Recomendadas

- Usar `LLAMA_CTX_SIZE=128000` y `LLAMA_PARALLEL=1` para modo profesor/RAG profundo.
- Para uso diario, permitir `LLAMA_CTX_SIZE=65536` si se busca menor memoria y latencia.
- Empezar con RAG textual + citas antes de embeddings.
- Anadir embeddings despues con una interfaz limpia.
- No activar Web por defecto.
- No permitir Web sin mostrar URLs y fecha/metadata cuando exista.
- Mantener el workspace como frontera de seguridad para documentos locales.

## Resultado Esperado

Al finalizar, el usuario podra:

- indexar una carpeta de documentacion;
- preguntar en lenguaje natural;
- recibir explicaciones con citas verificables;
- alternar entre Chat, RAG y Web desde el `+`;
- inspeccionar razonamiento/traza en un panel colapsable;
- usar el agente como profesor de documentacion local y buscador asistido.
