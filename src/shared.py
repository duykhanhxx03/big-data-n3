"""Frozen N3/N4 shared benchmark: pinned LeanDojo Benchmark 4 corpus/query contract (Data/extraction, Hoàng).

The validator checks an untouched copy of the bundle given to Nhóm 4 (n4_shared.zip). build_records and write_bundle
rebuild the same bytes from the raw LeanDojo release, so the bundle is reproducible without the zip.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Iterator

from extract.leandojo import _strip_attributes, statement_of

DATA_VERSION = 'leandojo-benchmark-4@29dcec07+split-v1.0'
SOURCE_COMMIT = '29dcec074de168ac2bf835a77ef68bbe069194c5'
POOL_ID = 'shared-frozen-pool'
EXPECTED_SHA256 = {
    'corpus.jsonl': 'c9157bd1381ca4147aa931f14b9985ac9bfe5cbfe892a982a24a133ff4221efd',
    'queries.jsonl': 'c2b764fafe981755ae7ca2614912498f0559317ea5427bcb7a7150cb45166004',
}
EXPECTED_COUNTS = {'corpus.jsonl': 180907, 'queries.jsonl': 898}
CORPUS_KEYS = {'id', 'name', 'statement', 'namespace', 'source_commit'}
QUERY_KEYS = {'id', 'statement', 'relevant_ids', 'split', 'pool_id'}

class ContractError(ValueError):
    """Source data violates the agreed frozen N3/N4 contract."""


def checksum(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for part in iter(lambda: stream.read(1024*1024), b''):
        digest.update(part)
    return digest.hexdigest()


def _iter_jsonl(raw: BinaryIO, label: str) -> Iterator[tuple[int,dict]]:
    for line_no, line in enumerate(raw, 1):
        if not line.strip():
            raise ContractError(f'{label}:{line_no}: blank line')
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError) as e:
            raise ContractError(f'{label}:{line_no}: invalid JSON/UTF-8: {e}') from e
        if not isinstance(value, dict):
            raise ContractError(f'{label}:{line_no}: expected object')
        yield line_no, value


def _keys(rec: dict, expected: set[str], label: str, lineno: int) -> None:
    if set(rec) != expected:
        raise ContractError(f'{label}:{lineno}: keys={sorted(rec)} expected={sorted(expected)}')


def theorem_id(query_id: str) -> str:
    # Query ID convention from original N3 exporter is 'Q' + theorem full_name.
    return query_id[1:] if query_id.startswith('Q') else query_id


def eligible(premise_id: str, query_id: str) -> bool:
    """Filter self-match BEFORE top_k. Other test theorems remain eligible."""
    return premise_id != theorem_id(query_id)


def load_and_validate(corpus_stream: BinaryIO, query_stream: BinaryIO, *, strict_counts: bool = True) -> dict:
    """Validate exact shared schema, referential integrity and frozen query pool.

    Returns provenance/quality report, not an altered/reconstructed dataset.
    Call after SHA-256 verification for official data.
    """
    ids: set[str] = set()
    corpus_count = 0
    for line_no, r in _iter_jsonl(corpus_stream,'corpus.jsonl'):
        _keys(r, CORPUS_KEYS, 'corpus.jsonl', line_no)
        if not all(isinstance(r[k], str) for k in CORPUS_KEYS):
            raise ContractError(f'corpus.jsonl:{line_no}: non-string field')
        if not r['id'] or not r['statement']:
            raise ContractError(f'corpus.jsonl:{line_no}: id or statement empty')
        if r['id'] != r['name']:
            raise ContractError(f'corpus.jsonl:{line_no}: name must equal full_name/id')
        if r['source_commit'] != SOURCE_COMMIT:
            raise ContractError(f'corpus.jsonl:{line_no}: unpinned source_commit')
        if r['id'] in ids:
            raise ContractError(f'corpus.jsonl:{line_no}: duplicate id {r["id"]}')
        ids.add(r['id'])
        corpus_count+=1
    if strict_counts and corpus_count != EXPECTED_COUNTS['corpus.jsonl']:
        raise ContractError(f'expected 180907 documents, got {corpus_count}')

    seen: set[str] = set()
    query_count = label_count = empty_relevance_queries = 0
    missing: Counter[str] = Counter()
    self_labels: list[str] = []
    for line_no, q in _iter_jsonl(query_stream,'queries.jsonl'):
        _keys(q, QUERY_KEYS, 'queries.jsonl', line_no)
        if not all(isinstance(q[k],str) for k in ('id','statement','split','pool_id')):
            raise ContractError(f'queries.jsonl:{line_no}: non-string field')
        if not isinstance(q['relevant_ids'],list) or not all(isinstance(p,str) for p in q['relevant_ids']):
            raise ContractError(f'queries.jsonl:{line_no}: relevant_ids must be list[str]')
        if not q['id'] or not q['statement']:
            raise ContractError(f'queries.jsonl:{line_no}: empty id/statement')
        if q['split'] != 'test' or q['pool_id'] != POOL_ID:
            raise ContractError(f'queries.jsonl:{line_no}: incorrect split or pool_id')
        if q['id'] in seen:
            raise ContractError(f'queries.jsonl:{line_no}: duplicate query id')
        if theorem_id(q['id']) not in ids:
            raise ContractError(f'queries.jsonl:{line_no}: theorem ID absent from corpus')
        seen.add(q['id'])
        if len(set(q['relevant_ids'])) != len(q['relevant_ids']):
            raise ContractError(f'queries.jsonl:{line_no}: duplicate relevant IDs')
        for p in q['relevant_ids']:
            if p not in ids:
                missing[p]+=1
            if not eligible(p, q['id']):
                self_labels.append(q['id'])
        query_count+=1
        if not q['relevant_ids']:
            empty_relevance_queries += 1
        label_count+=len(q['relevant_ids'])
    if strict_counts and query_count != EXPECTED_COUNTS['queries.jsonl']:
        raise ContractError(f'expected 898 queries, got {query_count}')
    if missing:
        raise ContractError(f'ground-truth IDs missing in corpus: {len(missing)}; samples={list(missing)[:5]}')
    if self_labels:
        raise ContractError(f'ground truth contains self-match for {len(self_labels)} queries; examples={self_labels[:5]}')
    return {'data_version':DATA_VERSION,'pool_id':POOL_ID,'mathlib_commit':SOURCE_COMMIT,
            'corpus_count':corpus_count,'query_count':query_count,
            'direct_premise_labels':label_count,
            'average_direct_premises_per_query':round(label_count/query_count,4) if query_count else 0,
            'self_labels':len(self_labels),'missing_premise_ids':len(missing),
            'queries_without_direct_premises':empty_relevance_queries,
            'schema':'N3/N4 frozen 5-field corpus/query contract',
            'eligible_rule':'exclude only own theorem before top_k; other test theorems allowed'}


def verify_directory(root: Path, *, strict_counts: bool = True) -> dict:
    root=Path(root)
    actual={}
    for name,expected in EXPECTED_SHA256.items():
        p=root/name
        if not p.is_file():
            raise ContractError(f'missing: {p}')
        with p.open('rb') as f:
            digest=checksum(f)
        actual[name]=digest
        if digest!=expected:
            raise ContractError(f'{name} SHA-256 mismatch: expected {expected}, actual {digest}')
    with (root/'corpus.jsonl').open('rb') as c,(root/'queries.jsonl').open('rb') as q:
        result=load_and_validate(c,q,strict_counts=strict_counts)
    return dict(result,checksums_sha256=actual,source='verified directory')


def verify_archive(path: Path, *, strict_counts: bool = True) -> dict:
    """Verify official N4 zip without extracting its ~66 MB into the repo."""
    path=Path(path)
    with zipfile.ZipFile(path) as z:
        names={Path(s).name:s for s in z.namelist() if not s.endswith('/')}
        actual={}
        for name,expected in EXPECTED_SHA256.items():
            if name not in names:
                raise ContractError(f'{name} missing in {path}')
            with z.open(names[name]) as f:
                actual[name]=checksum(f)
            if actual[name]!=expected:
                raise ContractError(f'{name} checksum mismatch: expected {expected}, actual {actual[name]}')
        with z.open(names['corpus.jsonl']) as corpus,z.open(names['queries.jsonl']) as queries:
            report=load_and_validate(corpus,queries,strict_counts=strict_counts)
        if 'checksums.txt' in names:
            raw=z.read(names['checksums.txt']).decode('utf-8')
            for name, digest in actual.items():
                if f'{digest}  {name}' not in raw:
                    raise ContractError(f'checksums.txt mismatch for {name}')
        return dict(report,checksums_sha256=actual,source=str(path))


def _traced_premises(theorem: dict) -> set[str]:
    """full_name of every premise annotated in any traced tactic of the theorem."""
    names: set[str] = set()
    for tactic in theorem.get('traced_tactics') or []:
        annotated = tactic.get('annotated_tactic') or []
        if isinstance(annotated, list) and len(annotated) >= 2 and isinstance(annotated[1], list):
            names.update(p['full_name'] for p in annotated[1] if p.get('full_name'))
    return names


def build_records(leandojo_dir: Path) -> tuple[list[dict], list[dict]]:
    """Rebuild the frozen corpus and query records from the raw LeanDojo Benchmark 4 release.

    Same rules as the original exporter (scripts/build_real_corpus.py and export_n4_shared.py at b1fbb35):
    corpus = every premise of corpus.jsonl, deduplicated by full_name (first occurrence wins), sorted by id;
    queries = random/test theorems with at least one annotated premise, statement = the theorem's corpus code,
    relevant_ids = annotated premises present in the corpus (may be empty), sorted by id.
    """
    root = Path(leandojo_dir)
    code: dict[str, str] = {}
    with (root / 'corpus.jsonl').open(encoding='utf-8') as f:
        for line in f:
            for p in json.loads(line)['premises']:
                code.setdefault(p['full_name'], p.get('code', ''))
    corpus = [{'id': n, 'name': n, 'statement': code[n], 'namespace': n.rsplit('.', 1)[0] if '.' in n else '',
               'source_commit': SOURCE_COMMIT} for n in sorted(code)]
    queries = []
    for thm in json.loads((root / 'random' / 'test.json').read_text(encoding='utf-8')):
        used = _traced_premises(thm)
        if not used:
            continue
        name = thm['full_name']
        queries.append({'id': 'Q' + name, 'statement': code.get(name, name),
                        'relevant_ids': sorted(p for p in used if p in code), 'split': 'test', 'pool_id': POOL_ID})
    queries.sort(key=lambda q: q['id'])
    return corpus, queries


def write_bundle(out_dir: Path, corpus: list[dict], queries: list[dict]) -> dict[str, str]:
    """Write corpus.jsonl, queries.jsonl and checksums.txt; return the SHA-256 of the two JSONL files.

    The frozen bytes have CRLF line endings because the original export ran on Windows in text mode, so the line
    ending is fixed here to give the same SHA-256 on every OS.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, rows in (('corpus.jsonl', corpus), ('queries.jsonl', queries)):
        with (out / name).open('w', encoding='utf-8', newline='\r\n') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with (out / name).open('rb') as f:
            digests[name] = checksum(f)
    lines = [f'# N3/N4 shared benchmark {DATA_VERSION}', f'# pool_id: {POOL_ID}',
             *(f'{d}  {n}' for n, d in digests.items())]
    (out / 'checksums.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    return digests


def has_proof_body(statement: str) -> bool:
    """True if a query statement carries the declaration body after ':=' (LeanDojo `code` keeps the proof of some
    theorems). Such a proof usually names the gold premises, so these queries leak their labels."""
    return len(statement_of(statement)) < len(_strip_attributes(statement).rstrip())
