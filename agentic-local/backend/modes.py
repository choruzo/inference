from __future__ import annotations

from pydantic import BaseModel

from backend.config import RAG_ENABLED, WEB_ENABLED


class ChatModes(BaseModel):
    chat: bool = True
    rag: bool = False
    web: bool = False
    reasoning_panel: bool = True

    def effective(self) -> "ChatModes":
        return self.model_copy(update={"rag": self.rag and RAG_ENABLED, "web": self.web and WEB_ENABLED})
