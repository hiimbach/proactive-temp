import asyncio

from openai import AsyncOpenAI


async def generate_n(
    prompt: str,
    n: int,
    temperature: float,
    port: int,
    model: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 1024,
) -> list[str]:
    """Generate N completions for a prompt using vLLM's OpenAI-compatible API.

    Uses the `n` parameter for efficient batched multi-sample generation.
    """
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")

    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return [choice.message.content or "" for choice in response.choices]


async def generate_batch(
    prompts: list[tuple[int, str]],
    n: int,
    temperature: float,
    port: int,
    model: str,
    max_concurrent: int = 32,
    max_tokens: int = 1024,
) -> dict[int, list[str]]:
    """Generate N completions for each prompt in a batch.

    Args:
        prompts: List of (index, prompt_text) tuples.
        n: Number of samples per prompt.
        temperature: Sampling temperature.
        port: vLLM server port.
        model: Model name for the API.
        max_concurrent: Max concurrent requests.
        max_tokens: Max tokens per generation.

    Returns:
        Dict mapping index -> list of N generation strings.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    async def _gen(idx: int, prompt: str):
        generations = await generate_n(prompt, n, temperature, port, model, semaphore, max_tokens)
        results[idx] = generations

    tasks = [_gen(idx, prompt) for idx, prompt in prompts]

    from tqdm.asyncio import tqdm_asyncio
    await tqdm_asyncio.gather(*tasks, desc=f"Generating (t={temperature}, n={n})")

    return results
