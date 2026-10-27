"""Verify and acquire model artifacts before accepting live input."""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=64)
def _digest(path, size, mtime_ns, ctime_ns):
    """Cache verified bytes only while the file's identity remains unchanged."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Model:
    name: str
    directory: str
    repository: str
    required: tuple[str, ...]
    patterns: tuple[str, ...] = ()
    legacy: tuple[str, ...] = ()
    revision: str | None = None
    checksums: tuple[tuple[str, str], ...] = ()

    def problems(self, root):
        """List missing artifacts and invalid metadata without loading weights."""
        directory = Path(root) / self.directory
        problems = []

        def check(path):
            if not path.is_file():
                problems.append(f'缺少文件：{path.name}')
            elif path.stat().st_size == 0:
                problems.append(f'空文件：{path.name}')
            elif path.suffix in {'.onnx', '.pt', '.safetensors', '.bin'} and path.stat().st_size < 1024:
                problems.append(f'权重文件过小：{path.name}')

        for pattern in self.required:
            files = list(directory.glob(pattern))
            if not files:
                problems.append(f'缺少文件：{pattern}')
            for path in files:
                check(path)
        for path in directory.glob('*.json'):
            try:
                data = json.loads(path.read_text())
                if path.name.endswith('.index.json'):
                    shards = data.get('weight_map', {})
                    if not shards:
                        problems.append(f'空权重索引：{path.name}')
                    for name in set(shards.values()):
                        check(directory / name)
            except (ValueError, OSError, AttributeError, TypeError):
                problems.append(f'无效 JSON/权重索引：{path.name}')
        for relative, expected in self.checksums:
            path = directory / relative
            if not path.is_file():
                problems.append(f'缺少文件：{relative}')
                continue
            try:
                stat = path.stat()
                actual = _digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if actual != expected:
                    problems.append(f'校验失败：{relative}')
            except OSError:
                problems.append(f'无法校验：{relative}')
        return list(dict.fromkeys(problems))

    def ready(self, root):
        """Return whether all required artifacts and metadata are present."""
        return not self.problems(root)

    def acquire(self, root, legacy_root):
        """Reuse complete existing weights, otherwise resume an explicit download."""
        root, legacy_root = Path(root), Path(legacy_root)
        target = root / self.directory
        problems = self.problems(root)
        if not problems:
            print(f'[model] 就绪：{self.name}', flush=True)
            return target
        for relative in self.legacy or (self.directory,):
            old = legacy_root / relative
            probe = replace(self, directory=relative)
            if not target.exists() and probe.ready(legacy_root):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(old.resolve(), target_is_directory=True)
                print(f'[model] 复用已有权重：{self.name} → {target}', flush=True)
                return target
        print(f'[model] 缺少或不完整：{self.name}；开始下载 {self.repository} → {target}', flush=True)
        from huggingface_hub import snapshot_download
        target.mkdir(parents=True, exist_ok=True)
        options = {}
        if self.revision:
            options['revision'] = self.revision
        if any(problem.startswith('校验失败：') for problem in problems):
            options['force_download'] = True
        try:
            snapshot_download(repo_id=self.repository, local_dir=str(target),
                              allow_patterns=list(self.patterns) or None,
                              ignore_patterns=['.git*', 'test_wavs/*', 'example/*', 'fig/*'],
                              **options)
        except Exception as exc:
            raise RuntimeError(f'{self.name} 下载失败（{type(exc).__name__}）；检查网络后重新运行，已下载内容会复用') from None
        problems = self.problems(root)
        if problems:
            raise RuntimeError(f'{self.name} 下载后仍不完整：{target}；' + '；'.join(problems))
        print(f'[model] 下载完成：{self.name}', flush=True)
        return target
