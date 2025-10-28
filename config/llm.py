"""
llm.py
Centralized configuration for LLM interactions (Ollama model selection, CLI behavior,
fallback preferences). Uses environment variables for runtime overrides.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import json

from utils.logger import logger
from . import constants

# --- LLM configuration constants ---
CONFIG_FILE = Path(__file__).parent / "llm_config.json"

# Model settings (will be loaded from config JSON if it exists)
_DEFAULT_CONFIG = {
    # Which model to use for each feature type
    "models": {
        "default": "gemma3:4b",
        "sentiment": "gemma3:4b",
        "intent": "llama2:latest",
    },
    # How to invoke each model via Ollama CLI
    "model_cli_modes": {
        "bge-m3:latest": "run",     # use 'ollama run bge-m3:latest "prompt"'
    "llama2:latest": "run",     # use 'ollama run llama2:latest "prompt"'
        "gemma3:4b": "run",         # use 'ollama run gemma3:4b "prompt"'
        "default": "run",           # fallback mode for other models
    },
    # Behavior flags
    "fallback": {
        "allow_local_fallback": True,  # whether to use local fallback scorers
        "fail_fast": False,           # if True, raise error instead of falling back
    }
}

class LLMConfig:
    def __init__(self):
        self._config = self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self) -> Dict:
        """Load settings from JSON if it exists, else use defaults."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    config = json.load(f)
                logger.debug(f"Loaded LLM config from {CONFIG_FILE}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load {CONFIG_FILE}: {e}. Using defaults.")
        return _DEFAULT_CONFIG.copy()
    
    def save(self):
        """Save current config back to JSON file."""
        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(self._config, f, indent=2)
            logger.debug(f"Saved LLM config to {CONFIG_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save {CONFIG_FILE}: {e}")

    def _apply_env_overrides(self):
        """Apply any environment-variable overrides to the config."""
        # MODEL overrides
        if model := os.environ.get("LLM_MODEL"):
            self._config["models"]["default"] = model
        # CLI mode overrides
        if mode := os.environ.get("LLM_CLI_MODE"):
            if mode in ("run", "generate"):
                self._config["model_cli_modes"]["default"] = mode
        # FALLBACK overrides
        if os.environ.get("FORCE_LLM_FALLBACK", "").lower() in ("1", "true", "yes"):
            self._config["fallback"]["allow_local_fallback"] = True
        if os.environ.get("LLM_FAIL_FAST", "").lower() in ("1", "true", "yes"):
            self._config["fallback"]["fail_fast"] = True

    def get_model(self, feature_type: Optional[str] = None) -> str:
        """Get the configured model name for a given feature type."""
        models = self._config["models"]
        if not feature_type or feature_type not in models or not models[feature_type]:
            return models["default"]
        return models[feature_type]

    def get_cli_mode(self, model: str) -> str:
        """Get the CLI mode (run/generate) for a given model."""
        modes = self._config["model_cli_modes"]
        return modes.get(model, modes["default"])
    
    def allow_fallback(self) -> bool:
        """Whether local fallback scorers are allowed."""
        return self._config["fallback"]["allow_local_fallback"]
    
    def fail_fast(self) -> bool:
        """Whether to fail immediately on LLM errors (vs. falling back)."""
        return self._config["fallback"]["fail_fast"]

    def build_cli_command(self, model: str, prompt: str, mode: Optional[str] = None) -> list:
        """Build the full Ollama CLI command for a given model and prompt.
        
        Args:
            model: Name of the model to use
            prompt: The input text or prompt
            mode: Optional override for CLI mode. If not provided, uses configured mode for model.
        """
        cli_mode = mode if mode is not None else self.get_cli_mode(model)
        if cli_mode == "embeddings":
            return ["ollama", cli_mode, model, "-f", prompt]  # -f for JSON format
        return ["ollama", cli_mode, model, prompt]

# Global singleton instance
config = LLMConfig()

# Save defaults if no config exists (helps users discover options)
if not CONFIG_FILE.exists():
    config.save()