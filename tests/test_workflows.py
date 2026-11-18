import yaml

from mousevision.config import REPO_ROOT

WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def load_workflow(name):
    with (WORKFLOWS / name).open() as fh:
        return yaml.safe_load(fh)


def triggers(workflow):
    # YAML 1.1 parses the key `on` as boolean True.
    return workflow.get("on", workflow.get(True))


def test_ci_runs_on_main_pushes_and_pull_requests():
    workflow = load_workflow("ci.yml")
    trig = triggers(workflow)

    assert trig["push"]["branches"] == ["main"]
    assert "pull_request" in trig


def test_ci_is_read_only():
    workflow = load_workflow("ci.yml")

    assert workflow["permissions"] == {"contents": "read"}


def test_mac_intel_check_is_manual_and_read_only():
    workflow = load_workflow("mac-intel-check.yml")
    trig = triggers(workflow)

    assert "workflow_dispatch" in trig
    assert workflow["permissions"] == {"contents": "read"}
