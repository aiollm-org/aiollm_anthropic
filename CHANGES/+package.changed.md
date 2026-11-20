**Breaking:** the Anthropic provider is its own package, `aiollm-anthropic`, imported
as `aiollm_anthropic`: `aiollm.providers.anthropic` becomes `aiollm_anthropic`, and each
of its modules keeps its name (`aiollm.providers.anthropic.types` becomes
`aiollm_anthropic.types`, and so on). `aiollm_anthropic.__version__` is its own
version.
