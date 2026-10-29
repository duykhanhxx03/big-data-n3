"""Contract tests for Hoàng data/extraction; do not require full benchmark."""
import io
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from shared import (ContractError,load_and_validate,eligible,theorem_id,SOURCE_COMMIT,POOL_ID,
                       build_records,write_bundle,has_proof_body)


def records(*data):
    return io.BytesIO((''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in data)).encode('utf-8'))


def corpus(name):
    return {'id':name,'name':name,'statement':'theorem true','namespace':'Mathlib','source_commit':SOURCE_COMMIT}


def query(rels):
    return {'id':'QMathlib.example','statement':'theorem true','relevant_ids':rels,'split':'test','pool_id':POOL_ID}

class SharedContractTest(unittest.TestCase):
    def test_valid(self):
        result=load_and_validate(records(corpus('Mathlib.example'),corpus('Mathlib.helper')),
               records(query(['Mathlib.helper'])),strict_counts=False)
        self.assertEqual(result['direct_premise_labels'],1)
    def test_self_not_eligible(self):
        self.assertEqual(theorem_id('QMathlib.example'),'Mathlib.example')
        self.assertFalse(eligible('Mathlib.example','QMathlib.example'))
        self.assertTrue(eligible('Mathlib.helper','QMathlib.example'))
    def test_self_ground_truth_rejected(self):
        with self.assertRaises(ContractError):
            load_and_validate(records(corpus('Mathlib.example')),records(query(['Mathlib.example'])),strict_counts=False)
    def test_missing_ground_truth_rejected(self):
        with self.assertRaises(ContractError):
            load_and_validate(records(corpus('Mathlib.example')),records(query(['Nonexistent'])),strict_counts=False)
    def test_bad_commit_rejected(self):
        bad=corpus('Mathlib.example');bad['source_commit']='different'
        with self.assertRaises(ContractError):
            load_and_validate(records(bad),records(query(['Mathlib.helper'])),strict_counts=False)
    def test_other_test_theorem_allowed(self):
        self.assertTrue(eligible('Mathlib.other_test_theorem','QMathlib.example'))
    def test_duplicate_document_rejected(self):
        with self.assertRaises(ContractError):
            load_and_validate(records(corpus('Mathlib.example'),corpus('Mathlib.example')),records(query(['Mathlib.example'])),strict_counts=False)


def tactic(*names):
    return {'annotated_tactic':['tac',[{'full_name':n} for n in names]]}


class RebuildTest(unittest.TestCase):
    """build_records/write_bundle reproduce the original exporter on a tiny LeanDojo-shaped release."""
    def setUp(self):
        import tempfile
        self.tmp=tempfile.TemporaryDirectory()
        root=Path(self.tmp.name)
        (root/'random').mkdir()
        files=[{'path':'Mathlib/B.lean','premises':[{'full_name':'b','code':'theorem b : B'},
                                                    {'full_name':'a','code':'theorem a : A := by\n  exact b'}]},
               {'path':'.lake/packages/lean4/A.lean','premises':[{'full_name':'b','code':'theorem b : core'}]}]
        (root/'corpus.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in files),encoding='utf-8')
        test=[{'full_name':'a','traced_tactics':[tactic('b','missing')]},
              {'full_name':'c','traced_tactics':[tactic('missing')]},
              {'full_name':'d','traced_tactics':[{'annotated_tactic':['rfl',[]]}]}]
        (root/'random'/'test.json').write_text(json.dumps(test),encoding='utf-8')
        self.root=root
    def tearDown(self):
        self.tmp.cleanup()
    def test_records(self):
        c,q=build_records(self.root)
        self.assertEqual([r['id'] for r in c],['a','b'])
        self.assertEqual(c[1]['statement'],'theorem b : B')                # first occurrence of a duplicate name wins
        self.assertEqual([r['id'] for r in q],['Qa','Qc'])                  # theorem without annotated premise dropped
        self.assertEqual(q[0]['relevant_ids'],['b'])                        # premise absent from corpus dropped
        self.assertEqual(q[1]['relevant_ids'],[])
        self.assertEqual(q[1]['statement'],'c')                             # theorem absent from corpus: name only
        load_and_validate(records(*c),records(q[0]),strict_counts=False)    # the validator rejects such a query
    def test_crlf_and_checksums(self):
        c,q=build_records(self.root)
        digests=write_bundle(self.root/'out',c,q)
        raw=(self.root/'out'/'corpus.jsonl').read_bytes()
        self.assertEqual(raw.count(b'\r\n'),2)
        self.assertNotIn(b'\n',raw.replace(b'\r\n',b''))
        text=(self.root/'out'/'checksums.txt').read_text(encoding='utf-8')
        for name,d in digests.items():
            self.assertIn(f'{d}  {name}',text)
    def test_proof_body(self):
        self.assertTrue(has_proof_body('theorem a : A := by\n  exact b'))
        self.assertFalse(has_proof_body('theorem a (h : x := 0) : A'))
        self.assertFalse(has_proof_body('/-- doc -/\n@[simp] theorem a : A'))

if __name__=='__main__':unittest.main()
