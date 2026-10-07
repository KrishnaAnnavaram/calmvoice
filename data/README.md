# data/

Git ignores everything in this folder except this file. Do not commit corpora, transcripts, audio or indexes.

## What the code needs

| File | Required? | Format |
|---|---|---|
| Corpus JSON | No. Without it, calmvoice uses the bundled demo corpus `src/calmvoice/data/corpus.json` | See the schema below |
| Red-team JSONL | No. `calmvoice redteam --out data/redteam.jsonl` generates a synthetic set | One object per line: `text`, `level` (`none`, `concern`, `crisis`), `category`, `template_id` |
| Retrieval queries JSONL | No. The bundled `src/calmvoice/data/retrieval_eval.jsonl` is the default | One object per line: `query`, `relevant` (list of `doc_id`) |

## Corpus schema

```json
{
  "corpus_id": "my-corpus",
  "version": "2026.10.1",
  "documents": [
    {
      "doc_id": "sleep-basics",
      "title": "Sleep basics",
      "topic": "sleep",
      "content_type": "psychoeducation",
      "source": "Publisher name",
      "url": "https://example.org/page",
      "license": "CC-BY-4.0",
      "text": "Unmodified text of the page ..."
    }
  ]
}
```

- `topic`: one of `stress`, `anxiety`, `low-mood`, `sleep`, `coping`, `help-seeking`, `relationships`.
- `content_type`: one of `psychoeducation`, `self-help-exercise`, `service-information`.
  Personal posts are not a valid content type.
- `license`: one of `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `OGL-UK-3.0`, `PUBLIC-DOMAIN`, `MIT`.
  The loader rejects every other value. Check the terms of each page before you add it.

Check a file with `calmvoice validate-corpus data/my_corpus.json`, then set `CALMVOICE_CORPUS=data/my_corpus.json`.

## Sources

- **Bundled demo corpus** (14 short documents): written for this repository, released as CC0-1.0.
  It is general wellbeing information. A clinician did not review it.
- **Your own corpus**: use public-health or charity psychoeducation pages whose licence allows reuse.
  For example, some government health pages use the Open Government Licence (`OGL-UK-3.0`). Read the
  licence of each page. Many health sites, books and PDFs do not allow redistribution.

## Data that calmvoice does not use

- **Forum and social-media posts** about mental health (for example the Kaggle dataset
  `neelghoshal/reddit-mental-health-data`). These are personal, sensitive texts by real people.
  calmvoice never uses them as a knowledge source and never shows them to users.
- **Commercial books and PDFs** without a reuse licence.
