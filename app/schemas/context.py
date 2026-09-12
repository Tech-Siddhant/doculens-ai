from pydantic import BaseModel, Field

class AssembledContext(BaseModel):
    """Result of context assembly, containing the formatted context and metadata."""
    context_text: str = Field(..., description="The fully assembled context string ready for the LLM")
    total_items: int = Field(..., description="Number of evidence items fully or partially included")
    truncated: bool = Field(default=False, description="Whether the context was truncated due to size limits")
