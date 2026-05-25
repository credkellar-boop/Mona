import torch

class MemoryBank:
    """
    Manages the sliding-window context to maintain narrative coherence 
    over long video generations.
    """
    def __init__(self, max_memory_size: int = 5):
        self.max_memory_size = max_memory_size
        self.states = []

    def add_state(self, hidden_state: torch.Tensor):
        """Adds a new state to the memory bank, popping the oldest if full."""
        if len(self.states) >= self.max_memory_size:
            self.states.pop(0)
        self.states.append(hidden_state.detach().cpu()) # Offload to CPU to save VRAM

    def get_context(self) -> torch.Tensor:
        """Retrieves the most recent context for the attention mechanism."""
        if not self.states:
            return None
        return torch.cat(self.states, dim=0).to("cuda")
