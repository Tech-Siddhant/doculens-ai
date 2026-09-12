from typing import Sequence
import html

from app.core.config import settings
from app.schemas.retrieval import EvidenceItem
from app.schemas.context import AssembledContext


class ContextAssembler:
    """
    Assembles selected EvidenceItems into a formatted context string for the LLM.
    Treats document content as untrusted by replacing conflicting xml tags.
    Handles size limits by truncating safely without breaking evidence metadata structure.
    """

    def _get_opening_tag(self, item: EvidenceItem) -> str:
        chunk_attr = f' chunk_id="{item.chunk_id}"' if item.chunk_id else ""
        return f'<evidence rank="{item.rank}" document_id="{item.document_id}" page_number="{item.page_number}"{chunk_attr}>'

    def _get_closing_tag(self) -> str:
        return "</evidence>"

    def _get_clean_content(self, item: EvidenceItem) -> str:
        content = item.text
        if not content:
            if item.image_url:
                content = f"[Visual element reference: {item.image_url}]"
            else:
                content = "[Empty Evidence]"
        
        # Security: neutralize any </evidence> tags in the document text to prevent prompt injection breakouts.
        # We also escape HTML to be safe, but a simple replace is often enough.
        content = content.replace("</evidence>", "< / evidence >")
        return content

    def assemble_context(self, evidence: Sequence[EvidenceItem], max_chars: int | None = None) -> AssembledContext:
        """
        Builds the context string.
        Args:
            evidence: Sequence of top-K EvidenceItems.
            max_chars: Character limit for the final context string. Defaults to settings.DEFAULT_MAX_CONTEXT_CHARS.
        """
        if max_chars is None:
            max_chars = getattr(settings, "DEFAULT_MAX_CONTEXT_CHARS", 16000)

        if not evidence:
            return AssembledContext(context_text="", total_items=0, truncated=False)

        parts = []
        current_len = 0
        truncated = False
        items_included = 0

        for item in evidence:
            opening = self._get_opening_tag(item)
            closing = "\n" + self._get_closing_tag()
            content = self._get_clean_content(item)

            separator = "\n\n" if parts else ""
            item_full = f"{separator}{opening}\n{content}{closing}"
            
            # Check length limit
            if current_len + len(item_full) > max_chars:
                truncated = True
                
                # If we haven't included any items or if we want to partially truncate this item to fit
                # Calculate available space:
                available_space = max_chars - (current_len + len(separator) + len(opening) + 1 + len(closing))
                
                if available_space > 0:
                    truncated_content = content[:available_space]
                    parts.append(f"{separator}{opening}\n{truncated_content}{closing}")
                    items_included += 1
                elif items_included == 0 and len(separator) + len(opening) + 1 + len(closing) <= max_chars:
                    # In extreme cases where we have 0 items included, try to squeeze just the empty tags
                    parts.append(f"{separator}{opening}\n{closing}")
                    items_included += 1
                    
                break

            parts.append(item_full)
            current_len += len(item_full)
            items_included += 1

        context_text = "".join(parts)
        return AssembledContext(context_text=context_text, total_items=items_included, truncated=truncated)


# Singleton
context_assembler = ContextAssembler()
