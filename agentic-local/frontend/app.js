const form = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt");
const sendButton = document.querySelector("#send");
const messagesEl = document.querySelector("#messages");
const statusEl = document.querySelector("#status");
const template = document.querySelector("#message-template");
const modeButton = document.querySelector("#mode-button");
const modeMenu = document.querySelector("#mode-menu");
const modeChips = document.querySelector("#mode-chips");
const clearChatButton = document.querySelector("#clear-chat");
const uploadInput = document.querySelector("#upload-input");
const uploadStatus = document.querySelector("#upload-status");
const MODE_KEY = "agentic-local-modes-v1";
const HISTORY_KEY = "agentic-local-history-v1";
const MAX_HISTORY_MESSAGES = 40;
const defaults = { chat: true, rag: false, web: false, reasoning_panel: true };
const LOADING_PHRASES = [
  "🔎 Rebuscando entre los pergaminos...",
  "🧭 Interrogando a los embeddings...",
  "🧩 Ordenando fragmentos sospechosamente relevantes...",
  "🤖 Convenciendo al modelo para que vaya al grano...",
  "📚 Poniendo las citas en fila...",
];

let history = loadHistory();
let modes = loadModes();

function loadModes() {
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(MODE_KEY) || "{}") };
  } catch (_) {
    return { ...defaults };
  }
}

function loadHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    if (!Array.isArray(stored)) return [];
    return stored
      .filter((item) => ["user", "assistant"].includes(item?.role) && typeof item?.content === "string")
      .slice(-MAX_HISTORY_MESSAGES);
  } catch (_) {
    return [];
  }
}

function saveHistory() {
  history = history.slice(-MAX_HISTORY_MESSAGES);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function renderModes() {
  modeMenu.querySelectorAll("[data-mode]").forEach((input) => {
    input.checked = Boolean(modes[input.dataset.mode]);
  });
  modeChips.replaceChildren();
  Object.entries(modes).forEach(([name, active]) => {
    if (!active) return;
    const chip = document.createElement("span");
    chip.className = "mode-chip";
    chip.textContent = name === "reasoning_panel" ? "Traza" : name.toUpperCase();
    modeChips.appendChild(chip);
  });
  localStorage.setItem(MODE_KEY, JSON.stringify(modes));
}

function addMessage(role, content, data = null) {
  const node = template.content.cloneNode(true);
  const article = node.querySelector(".message");
  article.classList.add(role);
  node.querySelector(".role").textContent = role === "assistant" ? "Agente" : "Tu";
  node.querySelector(".content").textContent = content;
  const citationsEl = node.querySelector(".citations");
  (data?.citations || []).forEach((citation) => {
    const link = document.createElement("a");
    link.className = "citation";
    const isWeb = citation.source_type === "web";
    link.href = isWeb ? citation.path : `/api/source?path=${encodeURIComponent(citation.path)}&line=${citation.start_line}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    const webMeta = [citation.provider, citation.section?.replace("Resultado web", "").replace(/^\s*·\s*/, "")]
      .filter(Boolean)
      .join(" · ");
    link.textContent = isWeb
      ? `[${citation.id}] Web · ${citation.title || new URL(citation.path).hostname}${webMeta ? ` · ${webMeta}` : ""}`
      : `[${citation.id}] ${citation.path}:${citation.start_line}-${citation.end_line}`;
    citationsEl.appendChild(link);
  });
  const details = node.querySelector(".trace-panel");
  if (data && modes.reasoning_panel && (data.trace?.length || data.reasoning || data.retrieval)) {
    const traceData = { reasoning: data.reasoning, trace: data.trace, retrieval: data.retrieval, stopped: data.stopped };
    details.hidden = false;
    details.querySelector("pre").textContent = JSON.stringify(traceData, null, 2);
    details.querySelector(".copy-trace").addEventListener("click", () => navigator.clipboard.writeText(JSON.stringify(traceData, null, 2)));
  }
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return article;
}

function startLoadingMessage() {
  let index = 0;
  const article = addMessage("assistant", LOADING_PHRASES[index]);
  article.classList.add("loading");
  article.setAttribute("aria-live", "polite");
  const content = article.querySelector(".content");
  const timer = window.setInterval(() => {
    index = (index + 1) % LOADING_PHRASES.length;
    content.textContent = LOADING_PHRASES[index];
  }, 1800);
  return () => {
    window.clearInterval(timer);
    article.remove();
  };
}

async function refreshStatus() {
  try {
    const [healthResponse, ragResponse] = await Promise.all([fetch("/api/health"), fetch("/api/rag/status")]);
    const health = await healthResponse.json();
    const rag = await ragResponse.json();
    statusEl.textContent = `${health.status} | ${rag.documents} docs | ${rag.chunks} chunks`;
  } catch (_) {
    statusEl.textContent = "Sin conexion con backend";
  }
}

modeButton.addEventListener("click", () => { modeMenu.hidden = !modeMenu.hidden; });
modeMenu.addEventListener("change", (event) => {
  const name = event.target.dataset.mode;
  if (!name) return;
  modes[name] = event.target.checked;
  if (name === "rag" && modes.rag) modes.chat = true;
  renderModes();
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".mode-picker")) modeMenu.hidden = true;
});

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files?.[0];
  if (!file) return;
  uploadStatus.textContent = `Subiendo ${file.name}...`;
  const body = new FormData();
  body.append("file", file);
  try {
    const response = await fetch("/api/rag/convert", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || response.statusText);
    const chunks = data.ingestion?.chunks ?? 0;
    uploadStatus.textContent = `${file.name}: ${chunks} fragmentos indexados.`;
    refreshStatus();
  } catch (error) {
    uploadStatus.textContent = `Error: ${error.message}`;
  } finally {
    uploadInput.value = "";
  }
});

clearChatButton.addEventListener("click", () => {
  history = [];
  localStorage.removeItem(HISTORY_KEY);
  messagesEl.replaceChildren();
  addMessage("assistant", "Chat limpiado.");
  promptInput.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = promptInput.value.trim();
  if (!message) return;
  addMessage("user", message);
  history.push({ role: "user", content: message });
  saveHistory();
  promptInput.value = "";
  sendButton.disabled = true;
  const stopLoading = startLoadingMessage();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: history.slice(0, -1), modes }),
    });
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    stopLoading();
    addMessage("assistant", data.answer, data);
    history.push({ role: "assistant", content: data.answer });
    saveHistory();
  } catch (error) {
    stopLoading();
    addMessage("assistant", `Error: ${error.message}`);
  } finally {
    sendButton.disabled = false;
    promptInput.focus();
  }
});

renderModes();
if (history.length) {
  history.forEach((item) => addMessage(item.role, item.content));
} else {
  addMessage("assistant", "Listo.");
}
refreshStatus();
