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

### Opciones

- Fase 1: usar un modelo de embeddings local servido por un proceso separado o una libreria Python.
- Fase 2: permitir backend configurable:
  - local embeddings;
  - endpoint OpenAI-compatible;
  - otro servidor HTTP local.

### Configuracion

Variables sugeridas:

```bash
EMBEDDINGS_PROVIDER=local
EMBEDDINGS_MODEL=local-embedding-model
EMBEDDINGS_BASE_URL=http://localhost:8081/v1
RAG_INDEX_DIR=/workspace/.rag_index
RAG_CHUNK_TOKENS=700
RAG_CHUNK_OVERLAP=100
RAG_TOP_K=8
RAG_RERANK_TOP_K=4
```

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

## Fases de Implementacion

### Fase 1: Base de modos y UI

- Extender `ChatRequest` con `modes`.
- Anadir menu `+` en el composer.
- Enviar modos activos al backend.
- Mostrar chips de modo.
- Crear panel colapsable para trace/razonamiento.
- Mantener compatibilidad con clientes antiguos sin `modes`.

### Fase 2: RAG local minimo

- Crear paquete `backend/rag`.
- Implementar ingestion para `.md` y `.txt`.
- Crear SQLite con `documents`, `chunks` y FTS5.
- Implementar busqueda textual.
- Anadir herramienta `rag_search`.
- Responder con citas basadas en chunks.

### Fase 3: Embeddings y busqueda hibrida

- Implementar cliente de embeddings.
- Guardar vectores versionados.
- Implementar vector search.
- Fusionar FTS + vector con RRF.
- Anadir pruebas de ranking.

### Fase 4: Reranking

- Implementar reranking por LLM con salida JSON.
- Reducir contexto final a top 3-5 chunks.
- Registrar scores y razones en trace.

### Fase 5: Web search/fetch

- Implementar `web_search`.
- Implementar `web_fetch`.
- Anadir modo Web en backend.
- Mostrar fuentes web separadas de fuentes locales.

### Fase 6: Calidad de profesor

- Prompts por tipo de respuesta:
  - explicacion paso a paso;
  - resumen;
  - comparacion;
  - tutorial;
  - diagnostico.
- Botones o comandos ligeros:
  - "explica mas simple";
  - "dame ejemplo";
  - "hazme preguntas";
  - "muestra fuentes".
- Evaluar respuestas contra documentos de prueba.

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
