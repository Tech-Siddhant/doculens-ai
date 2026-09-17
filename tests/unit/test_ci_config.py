"""Unit tests for Phase 8.9 CI/CD Pipeline Configuration."""

from pathlib import Path
import yaml


def test_ci_workflow_exists_and_valid() -> None:
    """Verify .github/workflows/ci.yml exists and has valid YAML syntax."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    ci_file = root_dir / ".github" / "workflows" / "ci.yml"

    assert ci_file.exists(), ".github/workflows/ci.yml must exist"

    content = ci_file.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)
    assert isinstance(parsed, dict), "ci.yml must parse to a valid YAML mapping"
    assert parsed.get("name") == "CI"


def test_ci_workflow_triggers_and_jobs() -> None:
    """Verify CI workflow triggers and mandatory backend/frontend jobs."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    ci_file = root_dir / ".github" / "workflows" / "ci.yml"
    parsed = yaml.safe_load(ci_file.read_text(encoding="utf-8"))

    # Triggers
    triggers = parsed.get("on") or parsed.get(True)  # PyYAML parses 'on' as boolean True
    assert triggers is not None
    if isinstance(triggers, dict):
        assert "push" in triggers
        assert "pull_request" in triggers

    # Jobs
    jobs = parsed.get("jobs", {})
    assert "backend" in jobs, "Backend job must be defined"
    assert "frontend" in jobs, "Frontend job must be defined"

    # Backend job details
    backend_job = jobs["backend"]
    assert backend_job.get("runs-on") == "ubuntu-latest"
    backend_steps = backend_job.get("steps", [])
    backend_step_runs = [s.get("run", "") for s in backend_steps if "run" in s]
    all_backend_runs = " ".join(backend_step_runs)

    assert "pip install" in all_backend_runs
    assert "pytest tests/unit" in all_backend_runs
    assert "scripts/run_evaluation.py" in all_backend_runs

    # Frontend job details
    frontend_job = jobs["frontend"]
    assert frontend_job.get("runs-on") == "ubuntu-latest"
    frontend_steps = frontend_job.get("steps", [])
    frontend_step_runs = [s.get("run", "") for s in frontend_steps if "run" in s]
    all_frontend_runs = " ".join(frontend_step_runs)

    assert "npm ci" in all_frontend_runs
    assert "npm run type-check" in all_frontend_runs
    assert "npm run lint" in all_frontend_runs
    assert "npm run test" in all_frontend_runs
    assert "npm run build" in all_frontend_runs


def test_ci_workflow_security_and_resource_constraints() -> None:
    """Verify CI workflow does not expose secrets or use prohibited heavy resources."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    ci_file = root_dir / ".github" / "workflows" / "ci.yml"
    content = ci_file.read_text(encoding="utf-8").lower()

    # No hardcoded secrets
    assert "aiver5" not in content
    assert "sk-" not in content
    assert "api_key=" not in content

    # Resource constraints
    assert "gpu" not in content
    assert "cuda" not in content
    assert "kubernetes" not in content
    assert "k8s" not in content
    assert "self-hosted" not in content


def test_live_evaluation_workflow() -> None:
    """Verify optional live evaluation workflow is dispatch-only and properly parameterized."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    eval_file = root_dir / ".github" / "workflows" / "live-evaluation.yml"

    assert eval_file.exists(), ".github/workflows/live-evaluation.yml must exist"

    content = eval_file.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)

    triggers = parsed.get("on") or parsed.get(True)
    assert "workflow_dispatch" in triggers, "Live eval must use workflow_dispatch"
    assert "push" not in triggers, "Live eval should not run automatically on push"
    assert "pull_request" not in triggers, "Live eval should not run automatically on PR"

    assert "${{ secrets.GEMINI_API_KEY }}" in content
    assert "run_evaluation.py" in content
