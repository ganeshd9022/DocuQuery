# llama_wrapper.py
# Thin wrapper that calls a compiled llama.cpp 'main' binary.
# Expects: ./main -m models/ggml-model.bin -p "PROMPT" -n 256
import subprocess, shlex, os, tempfile

LLAma_BIN = os.environ.get('LLAMA_CPP_BIN', './main')
MODEL_PATH = os.environ.get('LLAMA_CPP_MODEL', 'models/ggml-model.bin')

def generate_from_prompt(prompt: str, n_predict: int = 256, temp_prefix='tmp_prompt_'):
    # Write prompt to a temporary file and call llama.cpp with -f
    with tempfile.NamedTemporaryFile('w+', delete=False, prefix=temp_prefix, suffix='.txt') as tf:
        tf.write(prompt)
        tf.flush()
        tfname = tf.name
    cmd = f"{LLAma_BIN} -m {shlex.quote(MODEL_PATH)} -f {shlex.quote(tfname)} -n {n_predict} --temp 0.7 --top_k 40"
    try:
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=120)
        out = proc.stdout
        if proc.returncode != 0:
            return proc.stderr or out
        return out
    except Exception as e:
        return f"ERROR running model: {e}"
