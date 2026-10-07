"""The ``calmvoice`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace

from . import DISCLAIMER
from .companion import build_companion
from .config import Settings
from .corpus import load_corpus
from .embeddings import make_embedder
from .evaluation import evaluate_answers, evaluate_retrieval, evaluate_safety, level_counts, load_retrieval_set
from .index import build_index, load_or_build
from .memory import SessionMemory
from .retrieval import Retriever, RuleRewriter, SingleQuery
from .safety import SafetyGate, format_resources, load_resources
from .synthetic import make_redteam, read_jsonl, write_jsonl


def _settings(args) -> Settings:
    s = Settings.from_env()
    if getattr(args, "corpus", None):
        s = replace(s, corpus_path=args.corpus)
    if getattr(args, "index_dir", None):
        s = replace(s, index_dir=args.index_dir)
    if getattr(args, "embedder", None):
        s = replace(s, embedder=args.embedder)
    if getattr(args, "llm", None):
        s = replace(s, llm=args.llm)
    if getattr(args, "region", None):
        s = replace(s, region=args.region.upper())
    return s


def cmd_build_index(args) -> int:
    s = _settings(args)
    corpus = load_corpus(s.corpus_path or None)
    index, built = load_or_build(s.index_dir, corpus, make_embedder(s.embedder))
    state = "built" if built else "already current"
    print(f"index {state}: {s.index_dir} ({len(index.chunks)} chunks from {len(corpus.documents)} documents, "
          f"embedder {index.embedder_name}, corpus {corpus.fingerprint()})")
    return 0


def _print_reply(reply, as_json: bool) -> None:
    if as_json:
        print(json.dumps(reply.to_dict(), indent=2, ensure_ascii=False))
        return
    print(reply.text)
    if reply.sources:
        print("\nSources:")
        for src in reply.sources:
            print(f"  [{src.n}] {src.title} ({src.source}, {src.license})")
    print(f"\n(route: {reply.route}, risk: {reply.risk_level})")


def cmd_ask(args) -> int:
    companion = build_companion(_settings(args))
    _print_reply(companion.respond(args.message), args.json)
    return 0


def cmd_chat(args) -> int:
    s = _settings(args)
    companion = build_companion(s, memory=SessionMemory(enabled=args.memory))
    print(DISCLAIMER)
    print("Type a message. Type 'quit' to stop. Memory is", "ON" if args.memory else "OFF", "(never saved to disk).\n")
    while True:
        try:
            msg = input("you> ")
        except EOFError:
            break
        if msg.strip().lower() in {"quit", "exit"}:
            break
        _print_reply(companion.respond(msg), False)
        print()
    return 0


def cmd_redteam(args) -> int:
    items = make_redteam(per_template=args.per_template, seed=args.seed)
    path = write_jsonl(items, args.out)
    print(f"wrote {len(items)} synthetic messages to {path}: {level_counts(items)}")
    return 0


def cmd_eval_safety(args) -> int:
    items = read_jsonl(args.redteam) if args.redteam else make_redteam(seed=args.seed)
    report = evaluate_safety(SafetyGate(), items)
    out = {"rules": report.to_dict()}
    if args.with_classifier:
        from .safety_model import group_cross_validate

        out["classifier_group_cv"] = asdict(group_cross_validate(items, seed=args.seed))
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    lo, hi = report.crisis_recall_ci
    print(f"messages: {report.n} {level_counts(items)}")
    print(f"crisis recall (rules): {report.crisis_recall:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"crisis precision (rules): {report.crisis_precision:.3f}")
    print(f"false alarm rate on harmless messages: {report.false_alarm_rate:.3f}")
    print("confusion (true -> predicted):")
    for t, row in report.confusion.items():
        print(f"  {t:8s} {row}")
    print(f"missed crisis messages: {len(report.missed_crisis)}")
    for m in report.missed_crisis:
        print(f"  - {m}")
    if "classifier_group_cv" in out:
        print(f"learned classifier, GroupKFold by template: {out['classifier_group_cv']}")
    return 0


def cmd_eval_retrieval(args) -> int:
    s = _settings(args)
    corpus = load_corpus(s.corpus_path or None)
    emb = make_embedder(s.embedder)
    index = build_index(corpus, emb)
    queries = load_retrieval_set(args.queries)
    configs = [
        ("dense", Retriever(index, emb, mode="dense")),
        ("lexical", Retriever(index, emb, mode="lexical")),
        ("hybrid single-query", Retriever(index, emb, mode="hybrid", rewriter=SingleQuery())),
        ("hybrid multi-query", Retriever(index, emb, mode="hybrid", rewriter=RuleRewriter())),
    ]
    rows = [evaluate_retrieval(r, queries, k=args.k, label=name) for name, r in configs]
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return 0
    print(f"{len(queries)} queries, k={args.k}, embedder {emb.name}")
    print(f"{'configuration':22s} recall@k  hit@k  hit 95% CI      MRR")
    for r in rows:
        print(f"{r.mode:22s} {r.recall_at_k:8.3f} {r.hit_at_k:6.3f}  [{r.hit_ci[0]:.2f}, {r.hit_ci[1]:.2f}]  {r.mrr:6.3f}")
    return 0


def cmd_eval_answers(args) -> int:
    s = replace(_settings(args), index_dir="")
    companion = build_companion(s)
    rep = evaluate_answers(companion, load_retrieval_set(args.queries))
    print(json.dumps(asdict(rep), indent=2))
    return 0


def cmd_resources(args) -> int:
    res = load_resources(args.region or Settings.from_env().region)
    print(f"region: {res.region}")
    print(format_resources(res))
    return 0


def cmd_validate_corpus(args) -> int:
    try:
        corpus = load_corpus(args.path)
    except Exception as exc:  # pydantic.ValidationError or OSError
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {corpus.corpus_id} {corpus.version}, {len(corpus.documents)} documents, fingerprint {corpus.fingerprint()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="calmvoice", description="Safety-first wellbeing companion. " + DISCLAIMER)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--corpus", help="corpus JSON (default: bundled demo corpus)")
        sp.add_argument("--index-dir", help="index folder (default: CALMVOICE_INDEX_DIR)")
        sp.add_argument("--embedder", choices=["hashing", "minilm"])
        sp.add_argument("--llm", choices=["offline", "ollama"])
        sp.add_argument("--region", help="crisis line region, for example US, GB, IN")

    sp = sub.add_parser("build-index", help="build or reuse the persisted index")
    common(sp)
    sp.set_defaults(func=cmd_build_index)

    sp = sub.add_parser("ask", help="answer one message")
    common(sp)
    sp.add_argument("message")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("chat", help="interactive text chat")
    common(sp)
    sp.add_argument("--memory", action="store_true", help="keep the last turns in memory for this session")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("redteam", help="write the synthetic red-team set as JSONL")
    sp.add_argument("--out", required=True)
    sp.add_argument("--per-template", type=int, default=4)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=cmd_redteam)

    sp = sub.add_parser("eval-safety", help="crisis recall and false alarms on the red-team set")
    sp.add_argument("--redteam", help="JSONL from 'calmvoice redteam' (default: generate in memory)")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--with-classifier", action="store_true", help="also cross-validate the learned classifier")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_eval_safety)

    sp = sub.add_parser("eval-retrieval", help="recall@k and MRR for 4 retrieval configurations")
    common(sp)
    sp.add_argument("--queries", help="JSONL with query and relevant doc ids (default: bundled set)")
    sp.add_argument("--k", type=int, default=4)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_eval_retrieval)

    sp = sub.add_parser("eval-answers", help="citation validity and grounding of generated answers")
    common(sp)
    sp.add_argument("--queries")
    sp.set_defaults(func=cmd_eval_answers)

    sp = sub.add_parser("resources", help="show the crisis lines for a region")
    sp.add_argument("--region")
    sp.set_defaults(func=cmd_resources)

    sp = sub.add_parser("validate-corpus", help="check a corpus file against the schema and licence rules")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_validate_corpus)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
