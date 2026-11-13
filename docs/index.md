# Booster T1 manual

This robot package pins Booster T1 models and published controllers for Living
Proof Labs simulation and hallway verification. Start with the existing
[installation, reference checks, and configuration guide](../README.md).
It does not train controllers or certify real-world robot safety.

The package supplies an adapter to the separately owned simulator and verifier.
The guide describes both stock and Playground configurations and their known
reference-loop deviations. Read those limits before interpreting a result.

## Build and preview this manual

From the repository root, with Python 3.12:

```sh
python3 -m venv .cache/docs-venv
.cache/docs-venv/bin/pip install --require-hashes -r tools/docs/requirements.lock
python3 tools/docs/install-task.py
PATH="$PWD/.cache/docs-bin:$PATH" .cache/docs-venv/bin/python tools/docs/build.py
python3 -m http.server 8000 --directory artifacts/docs/html
```

Open http://localhost:8000/ and search for `playground`. This documentation build
never runs the simulation tasks or downloads robot models and controllers.
The dependency lock is the published guidelines v7.10.8 manual toolchain,
including its browser/test tools; only the renderer is used by this build.

## Related manuals

Use [Documentation home](https://docs.livingprooflabs.com/) to discover the
simulator, capability specifications, verifier, and reporter manuals. Their
responsibilities remain separate; this manual does not copy their contracts.
