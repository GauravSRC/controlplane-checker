"""Smoke test: the package and its submodules import cleanly.

Placeholder so ``pytest`` collects a passing suite on the fresh scaffold. Replace with
real detector/router/policy tests during implementation.
"""

import importlib

import pytest

MODULES = [
    "controlplane",
    "controlplane.config",
    "controlplane.proxy.app",
    "controlplane.proxy.capability",
    "controlplane.detectors.base",
    "controlplane.detectors.reflex",
    "controlplane.detectors.inspect",
    "controlplane.detectors.guard_model",
    "controlplane.detectors.groundedness",
    "controlplane.detectors.selfcheck",
    "controlplane.detectors.pii",
    "controlplane.detectors.adjudicate",
    "controlplane.policy.engine",
    "controlplane.router.decision",
    "controlplane.session.risk_ledger",
    "controlplane.ledger.otel",
    "controlplane.ledger.store",
    "controlplane.feedback.calibration",
    "controlplane.eval.generate_dataset",
    "controlplane.eval.harness",
    "controlplane.eval.optimizer",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
