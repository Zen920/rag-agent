from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent

def load_prompt(topic, txt_file):
    project_root = get_project_root()
    schema_path = project_root / 'agent' / 'prompts'/ topic / txt_file
    if not schema_path.exists():
        raise FileNotFoundError(f"Prompt directory not found: {schema_path}")
    try:
        with open(schema_path,  encoding="utf-8") as f: return f.read()
    except FileNotFoundError as e:
        raise e