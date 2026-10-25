"""Disposable provider probes for explicit desktop configuration changes."""
import asyncio
from contextlib import aclosing
import json
import os


async def probe_llm(role, provider):
    """Require a streamed text delta without user context or memory writes."""
    from studio.core.utils.llm.initialize import create, credential
    reply = create('Reply briefly to this connection check.', provider,
                   credential(provider, role), purpose=role)
    try:
        async with asyncio.timeout(20):
            async with aclosing(reply('Hello', '', [])) as deltas:
                async for delta in deltas:
                    if isinstance(delta, str) and delta.strip():
                        return
        raise RuntimeError('模型服务未输出流式文本。')
    finally:
        close = getattr(reply, 'aclose', None)
        if close:
            await close()


def validate_changed_llms(args):
    """Probe only LLM roles explicitly changed through the managed desktop UI."""
    roles = json.loads(os.environ.get('STUDIO_VALIDATE_SERVICES', '[]'))
    if not isinstance(roles, list) or any(role not in {'memory', 'reply'} for role in roles):
        raise ValueError('STUDIO_VALIDATE_SERVICES contains unknown roles')
    for role in dict.fromkeys(roles):
        provider = args.memory_llm if role == 'memory' else args.llm
        if provider == 'local':
            continue
        try:
            asyncio.run(probe_llm(role, provider))
        except Exception:
            raise RuntimeError(f'{role} 模型连接检查失败，请检查地址、模型、Key 和流式协议。') from None
