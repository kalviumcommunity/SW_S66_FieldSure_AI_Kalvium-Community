"""Workspace and environment configuration tests for FieldSure AI."""

from pathlib import Path
import re


def test_required_directories_exist():
    """Verify that all standard project directories exist."""
    repo_root = Path(__file__).resolve().parent.parent

    required_dirs = [
        "data",
        "src",
        "prompts",
        "outputs",
        "frontend",
        "tests",
    ]

    for dir_name in required_dirs:
        dir_path = repo_root / dir_name
        assert dir_path.is_dir(), f"Expected directory '{dir_name}' to exist at {dir_path}"


def test_env_example_contains_required_keys():
    """Verify that .env.example contains all mandatory configuration keys."""
    repo_root = Path(__file__).resolve().parent.parent
    env_example_path = repo_root / ".env.example"

    assert env_example_path.is_file(), ".env.example file is missing"

    content = env_example_path.read_text(encoding="utf-8")

    required_keys = [
        "API_BASE_URL",
        "API_KEY",
        "CHAT_MODEL",
        "EMBEDDING_MODEL",
        "VECTOR_DB_URL",
        "VECTOR_DB_PATH",
    ]

    for key in required_keys:
        pattern = rf"^\s*{re.escape(key)}\s*="
        assert re.search(pattern, content, re.MULTILINE), (
            f"Mandatory environment key '{key}' missing from .env.example"
        )


def test_gitignore_protects_secrets_and_data():
    """Verify that .gitignore excludes sensitive files, environments, and raw data."""
    repo_root = Path(__file__).resolve().parent.parent
    gitignore_path = repo_root / ".gitignore"

    assert gitignore_path.is_file(), ".gitignore file is missing"

    content = gitignore_path.read_text(encoding="utf-8")

    # Critical patterns that must be present
    assert ".env" in content, ".gitignore must exclude .env"
    assert ".venv" in content, ".gitignore must exclude .venv"
    assert "data/*" in content or "data/" in content, ".gitignore must exclude raw data"
    assert "__pycache__" in content, ".gitignore must exclude Python bytecode"


def test_pyproject_toml_configuration():
    """Verify that pyproject.toml defines required dependencies and python version."""
    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"

    assert pyproject_path.is_file(), "pyproject.toml is missing"

    content = pyproject_path.read_text(encoding="utf-8")

    assert "requires-python" in content
    expected_deps = ["pypdf", "python-docx", "tiktoken", "httpx", "python-dotenv"]
    for dep in expected_deps:
        assert dep in content, f"Expected dependency '{dep}' in pyproject.toml"


def test_env_parsing_and_no_real_secrets():
    """Verify that .env.example can be parsed safely and contains zero real secrets."""
    repo_root = Path(__file__).resolve().parent.parent
    env_example_path = repo_root / ".env.example"

    parsed = {}
    for line in env_example_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, v = stripped.split("=", 1)
            parsed[k.strip()] = v.strip()

    assert "API_BASE_URL" in parsed
    assert "API_KEY" in parsed
    assert "CHAT_MODEL" in parsed

    # Confirm placeholder only - never real credentials
    api_key = parsed["API_KEY"]
    assert "your_" in api_key or api_key == "", f"Detected potential secret in .env.example: {api_key}"


if __name__ == "__main__":
    test_required_directories_exist()
    test_env_example_contains_required_keys()
    test_gitignore_protects_secrets_and_data()
    test_pyproject_toml_configuration()
    test_env_parsing_and_no_real_secrets()
    print("All workspace configuration tests passed successfully!")


