const form = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt");
const sendButton = document.querySelector("#send");
const messagesEl = document.querySelector("#messages");
const statusEl = document.querySelector("#status");
const template = document.querySelector("#message-template");
const modeButton = document.querySelector("#mode-button");
const modeMenu = document.querySelector("#mode-menu");
const modeChips = document.querySelector("#mode-chips");
const MODE_KEY = "agentic-local-modes-v1";
const defaults = { chat: true, rag: false, web: false, reasoning_panel: true };

const history = [];
let modes = loadModes();

function loadModes() {
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(MODE_KEY) || "{}") };
  } catch (_) {
    return { ...defaults };
  }
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
    link.href = `/api/source?path=${encodeURIComponent(citation.path)}&line=${citation.start_line}`;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = `[${citation.id}] ${citation.path}:${citation.start_line}-${citation.end_line}`;
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = promptInput.value.trim();
  if (!message) return;
  addMessage("user", message);
  history.push({ role: "user", content: message });
  promptInput.value = "";
  sendButton.disabled = true;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: history.slice(-12, -1), modes }),
    });
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    addMessage("assistant", data.answer, data);
    history.push({ role: "assistant", content: data.answer });
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`);
  } finally {
    sendButton.disabled = false;
    promptInput.focus();
  }
});

renderModes();
addMessage("assistant", "Listo.");
refreshStatus();
