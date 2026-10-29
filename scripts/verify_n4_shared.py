#!/usr/bin/env python3
"""Verify untouched N3/N4 frozen dataset from zip or directory."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shared import ContractError,verify_archive,verify_directory


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('input',nargs='?',default=str(ROOT/'data'/'n4_shared'),
                   help='Path to n4_shared.zip or directory with corpus.jsonl and queries.jsonl (default: data/n4_shared)')
    p.add_argument('--report',default='results/data_quality/shared_validation.json')
    args=p.parse_args()
    try:
        inp=Path(args.input)
        result=verify_archive(inp) if inp.is_file() else verify_directory(inp)
    except (ContractError,FileNotFoundError,ValueError,OSError) as e:
        print('DATA CONTRACT FAILED:',e,file=sys.stderr)
        return 1
    dest=ROOT/args.report
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print('Report:',dest)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
