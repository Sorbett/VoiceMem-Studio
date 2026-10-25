"""Validate startup-only component selections before loading model code."""
from __future__ import annotations

from functools import lru_cache
import importlib
import json
import math
import os
import re
from urllib.parse import urlsplit


@lru_cache(maxsize=1)
def catalog():
    """Read the shared desktop/backend component descriptions without model imports."""
    from studio.paths import STUDIO
    return json.loads((STUDIO / 'apps/component-catalog.json').read_text(encoding='utf-8'))


def endpoint(value):
    """Validate an HTTP API root without credentials, query parameters or fragments."""
    value = str(value or '').strip().rstrip('/')
    url = urlsplit(value)
    local = url.hostname == 'localhost' or url.hostname == '::1'
    if url.hostname:
        import ipaddress
        try:
            local = local or ipaddress.ip_address(url.hostname).is_loopback
        except ValueError:
            pass
    if (url.scheme not in {'https', 'http'} or not url.hostname
            or url.username or url.password or url.query or url.fragment
            or (url.scheme == 'http' and not local)):
        raise ValueError('服务地址需要 HTTPS；本机服务可使用 HTTP，地址不能包含凭据或查询参数。')
    try:
        url.port
    except ValueError:
        raise ValueError('服务端口无效。') from None
    return value


def validate(role, value):
    """Normalize one binding; reject unknown fields and unsafe/malformed options."""
    if role not in catalog() or not isinstance(value, dict):
        raise ValueError('组件配置格式不正确。')
    unknown = set(value) - {'provider', 'model', 'baseUrl', 'apiKey', 'voice', 'language', 'factory', 'options'}
    if unknown:
        raise ValueError('组件包含未知配置字段。')
    provider = value.get('provider', 'recommended')
    if provider not in catalog()[role]['providers']:
        raise ValueError(f'{role} 不支持此接入方式。')
    if provider == 'recommended':
        return {'provider': provider}
    defaults = catalog()[role]['providers'][provider]
    result = {'provider': provider}
    for name in ('model', 'baseUrl', 'apiKey', 'voice', 'language', 'factory'):
        item = value.get(name, defaults.get(name, ''))
        if not isinstance(item, str) or len(item) > (8192 if name == 'apiKey' else 1000):
            raise ValueError(f'{role}: {name} 格式不正确。')
        item = item.strip()
        if any(ord(char) < 32 or ord(char) == 127 for char in item):
            raise ValueError(f'{role}: {name} 不能包含控制字符。')
        result[name] = item
    if result['baseUrl']:
        result['baseUrl'] = endpoint(result['baseUrl'])
    if provider == 'deepgram':
        if not all(result[name] for name in ('model', 'baseUrl', 'language')):
            raise ValueError('识别 API 需要模型、服务地址和语言。')
        return result
    if provider == 'openai':
        if not all(result[name] for name in ('model', 'baseUrl', 'voice')):
            raise ValueError('语音 API 需要模型、服务地址和音色。')
        return result
    if not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', result['factory']):
        raise ValueError('适配器填写已安装的 Python 模块，例如 my_adapter:create。')
    options = value.get('options', {})
    if not isinstance(options, dict):
        raise ValueError('适配器参数必须为 JSON 对象。')
    def check(item, depth=0):
        if depth > 8:
            raise ValueError('适配器参数嵌套过深。')
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str) or re.fullmatch(
                        r'api[_-]?key|access[_-]?token|token|secret|password|authorization|credentials', key, re.I):
                    raise ValueError('凭据请填写在 API Key 中，不要放在适配器参数里。')
                check(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                check(child, depth + 1)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError('适配器参数不能包含非有限数值。')
        elif not isinstance(item, (str, int, float, bool, type(None))):
            raise ValueError('适配器参数必须是 JSON 数据。')
    check(options)
    if len(json.dumps(options, ensure_ascii=False)) > 16384:
        raise ValueError('适配器参数过长。')
    result['options'] = options
    return result


@lru_cache(maxsize=8)
def _selections(raw):
    try:
        values = json.loads(raw)
    except (TypeError, ValueError):
        raise ValueError('STUDIO_COMPONENTS 必须为有效的 JSON 对象。') from None
    if not isinstance(values, dict) or set(values) - set(catalog()):
        raise ValueError('STUDIO_COMPONENTS 包含未知组件。')
    if any(isinstance(value, dict) and value.get('apiKey') for value in values.values()):
        raise ValueError('组件凭据请通过 STUDIO_<ROLE>_API_KEY 设置。')
    return {role: validate(role, values.get(role, {})) for role in catalog()}


def selection(role):
    """Resolve a binding; role-specific credentials stay outside the public config."""
    value = dict(_selections(os.environ.get('STUDIO_COMPONENTS', '{}'))[role])
    value['apiKey'] = os.environ.get(f'STUDIO_{role.upper()}_API_KEY', '')
    return value


def public_selections():
    """Return non-secret configuration and contract descriptions for settings."""
    values = {}
    for role in catalog():
        value = selection(role)
        key = value.pop('apiKey', '')
        value['hasKey'] = bool(key)
        value['configured'] = value['provider'] in {'recommended', 'adapter'} or bool(key)
        values[role] = value
    return values


def custom(role, value=None):
    """Construct an explicitly selected, installed adapter without loading scripts.

    Factories receive keyword arguments ``model``, ``base_url``, ``api_key`` and
    ``options``. They must return the existing role contract synchronously.
    Native scheduling and cancellation belong to the adapter implementation.
    """
    value = selection(role) if value is None else value
    module, name = value['factory'].split(':')
    factory = getattr(importlib.import_module(module), name)
    adapter = factory(model=value['model'], base_url=value['baseUrl'],
                      api_key=value['apiKey'], options=json.loads(json.dumps(value['options'])))
    from .contracts import checked
    result = checked(role, adapter)
    if role == 'speech':
        import hashlib
        identity = {key: value[key] for key in ('factory', 'model', 'baseUrl', 'voice', 'options')}
        identity['voice_id'] = str(getattr(adapter, 'cache_voice_id', ''))
        result.cache_voice_id = 'adapter:' + hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return result


def vad():
    """Keep VoiceMem's plain-function factory contract while sharing the detector."""
    return _vad()


@lru_cache(maxsize=1)
def _vad():
    """Return the recommended detector or a checked stream-isolated adapter."""
    value = selection('vad')
    if value['provider'] == 'adapter':
        return custom('vad', value)
    from voicemem.utils.audio.stream_io import make_vad
    return make_vad()
