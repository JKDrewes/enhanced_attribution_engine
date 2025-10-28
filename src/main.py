"""
main.py
Orchestrates the backend workflow:
1. Cleans previous outputs
2. Pre-processes data
3. LLM feature generation (sentiment + intent)
4. Logistic regression model training
5. Shapley attribution calculation
6. Model evaluation
7. Reporting (visualizations + narrative)
"""

# Ensure bootstrap is importable even when running from the script's folder
import sys
from pathlib import Path
try:
    import bootstrap
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import bootstrap
from utils.logger import logger
from utils.paths import (
    PROCESSED_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    LOGS_DIR
)
import subprocess
import sys
import shutil
from pathlib import Path
import os

def clean_previous_outputs():
    """Deletes previous outputs to ensure fresh pipeline run."""
    logger.info("Cleaning previous outputs...")

    # Close any open logger handlers so log files can be removed on Windows
    try:
        from utils.logger import close_logger
        close_logger()
    except Exception:
        logger.debug("Unable to close logger handlers before cleanup; continuing.")

    folders_to_clean = [PROCESSED_DIR, OUTPUTS_DIR, REPORTS_DIR]
    for folder in folders_to_clean:
        if folder.exists():
            for item in folder.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    logger.warning(f"Failed to delete {item}: {e}")

    # Clear logs
    if LOGS_DIR.exists():
        for log_file in LOGS_DIR.glob("*.log"):
            try:
                # Attempt removal with a short retry if file is locked by the OS
                try:
                    log_file.unlink()
                except PermissionError:
                    # Try once more after forcing garbage collection and small sleep
                    import time, gc
                    gc.collect()
                    time.sleep(0.1)
                    try:
                        log_file.unlink()
                    except Exception as e2:
                        logger.warning(f"Failed to delete log {log_file}: {e2}")
            except Exception as e:
                logger.warning(f"Failed to delete log {log_file}: {e}")

    # Reopen logger so further steps continue logging normally
    try:
        from utils.logger import reopen_logger
        reopen_logger()
    except Exception:
        logger.debug("Unable to re-open logger after cleanup; continuing.")

    logger.info("Previous outputs cleaned.")

def run_script(script_path: str):
    """Utility to run a Python script and log errors if any"""
    logger.info(f"Running {script_path} ...")
    try:
        # Ensure subprocess knows about both project root and src/ so imports
        # resolve when running files as scripts. Prefer running as modules
        # (python -m pkg.module) when possible.
        env = os.environ.copy()
        # Set UTF-8 encoding for subprocess input/output
        env["PYTHONIOENCODING"] = "utf-8"
        # Force UTF-8 usage on Windows
        if os.name == 'nt':  # Windows
            env["PYTHONUTF8"] = "1"
        existing = env.get("PYTHONPATH", "")
        root_path = str(bootstrap.ROOT_DIR)
        src_path = str(bootstrap.SRC_DIR)
        paths = os.pathsep.join(p for p in (root_path, src_path) if p)
        env["PYTHONPATH"] = paths + (os.pathsep + existing if existing else "")

        # If provided a file under src/, prefer running it as a module (python -m)
        spath = Path(script_path)
        if spath.exists() and spath.suffix == ".py":
            # Convert to module path: src/attribution_model/foo.py -> src.attribution_model.foo
            module = spath.with_suffix("").as_posix().replace('/', '.')
            subprocess.run([sys.executable, "-m", module], 
                         check=True, 
                         env=env,
                         encoding='utf-8',
                         errors='replace',
                         text=True)  # Ensures text mode with encoding
        else:
            subprocess.run([sys.executable, script_path], 
                         check=True, 
                         env=env,
                         encoding='utf-8',
                         errors='replace',
                         text=True)  # Ensures text mode with encoding
        logger.info(f"{script_path} completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")

def main():
    logger.info("=== STARTING ATTRIBUTION ENGINE PIPELINE ===")

    # --- Step 0: Clean previous outputs ---
    clean_previous_outputs()

    # --- Step 1: Data Processing ---
    logger.info("=== STEP 1: DATA PROCESSING ===")
    run_script("src/data_processing/clean_data.py")

    # --- Step 2: LLM Pipeline ---
    logger.info("=== STEP 2: LLM PIPELINE ===")
    
    # Check LLM availability
    from config.llm import config as llm_config
    def check_ollama():
        """Quick health check for the Ollama CLI."""
        import shutil, subprocess
        if not llm_config.allow_fallback():
            if not shutil.which("ollama"):
                logger.error("Ollama CLI not found and fallback is disabled.")
                raise RuntimeError("Ollama CLI not found and fallback is disabled")
            return True

        if os.environ.get("FORCE_LLM_FALLBACK", "").lower() in ("1", "true", "yes"):
            logger.info("FORCE_LLM_FALLBACK set — LLM steps will use local fallbacks.")
            return False

        if not shutil.which("ollama"):
            logger.warning("Ollama CLI not found on PATH. LLM steps will use local fallbacks.")
            return False

        try:
            r = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
            model = llm_config.get_model()
            logger.info(f"Testing Ollama with model {model}...")
            test_result = subprocess.run(
                llm_config.build_cli_command(model, "Say 'ok'."),
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
            logger.info(f"Ollama active: {r.stdout.strip()}")
            return True
        except Exception as e:
            msg = f"Ollama health check failed: {e}"
            if llm_config.fail_fast():
                raise RuntimeError(msg)
            logger.warning(f"{msg}. LLM steps will fall back on error.")
            return False

    # Run LLM health check
    ollama_available = check_ollama()
    
    # Run LLM analysis pipelines
    logger.info("Running sentiment analysis...")
    run_script("src/llm_engine/sentiment_analysis.py")
    
    logger.info("Running intent detection...")
    run_script("src/llm_engine/intent_detection.py")
    
    logger.info("Exporting LLM features...")
    run_script("src/llm_engine/feature_export.py")

    # --- Step 3: Model Training & Attribution ---
    logger.info("=== STEP 3: MODEL TRAINING & ATTRIBUTION ===")
    
    logger.info("Training classifier model...")
    run_script("src/attribution_model/classifier_model.py")
    
    logger.info("Calculating Shapley attributions...")
    run_script("src/attribution_model/shapely_attribution_model.py")

    # --- Step 4: Evaluation ---
    logger.info("=== STEP 4: EVALUATION ===")
    
    logger.info("Running attribution model evaluation...")
    run_script("src/attribution_model/evaluation.py")
    
    logger.info("Running LLM evaluation...")
    run_script("src/llm_engine/evaluation.py")

    # --- Step 5: Reporting & Visualization ---
    logger.info("=== STEP 5: REPORTING & VISUALIZATION ===")
    
    logger.info("Generating visualizations...")
    run_script("src/reporting/visualization.py")
    
    logger.info("Generating narrative summary...")
    run_script("src/reporting/narrative_summary.py")
    
    logger.info("Generating final report...")
    run_script("src/reporting/generate_report.py")

    logger.info("=== ATTRIBUTION ENGINE PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
