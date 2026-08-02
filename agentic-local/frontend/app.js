const form = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt");
const sendButton = document.querySelector("#send");
const messagesEl = document.querySelector("#messages");
const statusEl = document.querySelector("#status");
const template = document.querySelector("#message-template");

const history = [];

function addMessage(role, content) {
  const node = template.content.cloneNode(true);
  const article = node.querySelector(".message");
  article.classList.add(role);
  node.querySelector(".role").textContent = role;
  node.querySelector(".content").textContent = content;
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    statusEl.textContent = `${data.status} | ${data.workspace}`;
  } catch (error) {
    statusEl.textContent = "Sin conexion con backend";
  }
}

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
      body: JSON.stringify({ message, history: history.slice(-12, -1) }),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = await response.json();
    const trace = data.trace ? `\n\n[stopped: ${data.stopped}, steps: ${data.trace.length}]` : "";
    addMessage("assistant", `${data.answer}${trace}`);
    history.push({ role: "assistant", content: data.answer });
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`);
  } finally {
    sendButton.disabled = false;
    promptInput.focus();
  }
});

addMessage("assistant", "Listo. Puedo inspeccionar y editar archivos dentro del workspace montado.");
refreshStatus();
