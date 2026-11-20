# aiollm-anthropic

Anthropic's Messages and Files APIs for [aiollm](https://github.com/aiollm-org/aiollm), the async-first toolkit for typed HTTP API clients.

`AnthropicClient` sends Anthropic's own methods (`CreateMessage`, `StreamMessage` and the Files methods) with types generated from Anthropic's official SDK types and its Files API reference, and has aiollm's provider-neutral `client.generation` and `client.files`, so code written for any aiollm provider runs on it.

> **Status: pre-alpha.** The package was part of aiollm up to 0.14.0 as `aiollm.providers.anthropic`; it is now `aiollm_anthropic`. Its public API may still change: before 1.0 a breaking change bumps the minor version.

## Installation

The package requires Python 3.13 or newer. It is not on PyPI yet, so install it from source; the core, [aiollm-core](https://pypi.org/project/aiollm-core/), comes from PyPI:

```bash
git clone https://github.com/aiollm-org/aiollm_anthropic.git
cd aiollm_anthropic
uv sync
```

Or with pip: `pip install -e .`

## Quick start

`AnthropicClient()` reads the key from `ANTHROPIC_API_KEY`; pass a token to give it yourself.

```python title="quick_start.py"
from aiollm_anthropic import AnthropicClient


async def ask() -> str:
    async with AnthropicClient() as client:
        response = await client.generation.create(
            model="claude-opus-5-5",
            input="Hi",
            instructions="Be brief",
            max_output_tokens=256,
        )
    return response.text
```

Run it with `asyncio.run(ask())`.

## Documentation

The guide and the API reference are on the aiollm site, **[aiollm-org.github.io/aiollm](https://aiollm-org.github.io/aiollm/)**: [provider clients](https://aiollm-org.github.io/aiollm/guide/providers/), [shared generation](https://aiollm-org.github.io/aiollm/guide/generation/) and [files](https://aiollm-org.github.io/aiollm/guide/files/). The changes up to 0.14.0 are in [aiollm's changelog](https://github.com/aiollm-org/aiollm/blob/master/CHANGELOG.md); the later ones in [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
