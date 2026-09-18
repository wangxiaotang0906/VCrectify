# Contributing

Install the framework and test tools with `python -m pip install -e ".[test]"`, then run `python -m pytest -q`. The default test suite uses no external model service or downloaded dataset. Tests of optional official backbones are skipped unless their source and dependencies are present. Never store credentials in a test fixture or config.

A backbone contribution should provide its input/output contract, upstream source/version/license, a minimal configuration and tests for axis alignment, fresh initialization, selected-only correction and checkpoint roundtrips. Keep numerical architectures in adapters; acquisition, reliability updates and label access belong in the engine.

Method changes need a formula-level explanation and a test of the scientific invariant being changed. Particularly sensitive invariants are prediction sealing before observation, LOPO isolation, condition-balanced replay, and evaluation having no effect on acquisition. Do not write tests that only reproduce the implementation without testing its consequence.

Label toy or mocked results clearly. Experimental claims need source data provenance, split manifests, seed lists, complete configurations, outcome metrics and class support. Do not replace a failed LLM response with a guessed label. Keep source datasets, restricted upstream code, weights and run caches out of pull requests.

The project's independent code is Apache-2.0. A contribution must be compatible with that license; upstream model/data terms remain separate. Keep software citation metadata current, add final manuscript metadata when available, and review release archive contents before each release.
