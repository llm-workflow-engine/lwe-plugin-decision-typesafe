# LLM Workflow Engine (LWE) TypeSafe Decision Provider plugin

TypeSafe Decision Provider plugin for [LLM Workflow Engine](https://github.com/llm-workflow-engine/llm-workflow-engine).

Access to [TypeSafe's Jev](https://typesafe.ai/) decision models through the
[LangChain TypeSafe integration](https://docs.langchain.com/oss/python/integrations/providers/typesafe).
Jev evaluates typed questions and returns probabilities, choices, and scores.

## Installation

### Export API key

Create an API key with [TypeSafe](https://typesafe.ai/) and export it:

```bash
export TYPESAFE_API_KEY=<API_KEY>
```

### From packages

Install the latest version directly from GitHub with pip:

```bash
pip install git+https://github.com/llm-workflow-engine/lwe-plugin-decision-typesafe
```

### From source (recommended for development)

```bash
git clone https://github.com/llm-workflow-engine/lwe-plugin-decision-typesafe.git
cd lwe-plugin-decision-typesafe
pip install -e .
```

The plugin pins the
prerelease `langchain-typesafe==0.0.1a3` integration, whose typed interface is
covered by its tests.

## Configuration

Add a decision preset to `config.yaml` in your LWE profile:

```yaml
decisions:
  presets:
    jev-triage:
      provider: typesafe
      model: jev-latest
      timeout: 30.0
      max_retries: 2
```

`api_key_env_var` can select a different credential environment variable.
The native wrapper honors `TYPESAFE_BASE_URL` for compatible System One
endpoints and appends `/v1/systemone`. No provider-specific `options` are needed.

The plugin registers `typesafe` under `lwe_decision_providers`. It does not
require `plugins.enabled` configuration or appear in the chat CLI.

## Usage

### Python

```python
from lwe.decisions import DecisionRequest, DecisionService, DecisionSettings, Noul

settings = DecisionSettings.from_preset("jev-triage")
service = DecisionService(settings)
result = service.decide(DecisionRequest(
    state={"message": "Checkout is unavailable."},
    questions={"urgent": Noul(instructions="Does this need urgent attention?")},
))
print(result.model_dump(mode="json"))
```

Alternatively construct `DecisionSettings(provider="typesafe", model="jev-latest")`
directly. `from_preset("jev-triage", profile="production")` selects a preset from
the `production` LWE profile; omitting `profile` uses `default`.

### Ansible

```yaml
- name: Classify a support request
  lwe_decide:
    preset: jev-triage
    state: "{{ support_request }}"
    questions:
      urgent:
        type: noul
        instructions: Does this require urgent attention?
  register: triage
```

Use `triage.answers.urgent.noul` in your workflow's threshold and fallback logic.
See the [LWE decision API documentation](https://github.com/llm-workflow-engine/llm-workflow-engine/blob/main/docs/decisions.rst)
for result types, retries, check mode, and workflow configuration.

The adapter preserves typed answers, confidence, distributions, token usage,
and request IDs, and closes its synchronous HTTP client after each call.
The current wrapper accepts text and structured state; media fields are
rejected before any download or API call. If LangSmith tracing is enabled,
the native wrapper can record classification inputs and outputs.
