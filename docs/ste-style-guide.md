# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

This section gives the technical names and the technical verbs of calmvoice. The README uses each term with only this meaning.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **companion** | The `Companion` pipeline that answers one message | bot, chatbot, assistant, agent |
| **message** | The text that the user types or says in one turn | query (except in retrieval), question, prompt |
| **reply** | The `Reply` object that the companion returns | response, output, answer (for the object) |
| **answer** | A reply with the `answer` route: text from the generator with citations | solution, advice |
| **route** | The `route` field of a reply: `crisis`, `out_of_scope`, `answer`, `blocked`, `empty` | path, branch, outcome |
| **risk level** | The `RiskLevel` of a message: `none`, `concern`, `crisis` | severity, danger score |
| **safety gate** | `SafetyGate`: the keyword rules plus the optional classifier | filter, moderation, guardrail |
| **rule** | One regular expression in `safety.py` with a category and a risk level | keyword list, trigger |
| **classifier** | The optional `LearnedRiskClassifier` (TF-IDF and logistic regression) | model (alone), detector (for the classifier) |
| **escalation message** | The fixed reply for a crisis message, with crisis lines | hotline script, emergency answer |
| **crisis line** | One telephone or text service in `resources.json` | hotline, helpline (except in a product name) |
| **scope limit** | The check that refuses diagnosis and medication questions | topic filter, policy check |
| **output guard** | `guard_output`: the check on generated text before the user sees it | post-filter, validator |
| **corpus** | The validated JSON file of curated documents | knowledge base, dataset, database |
| **document** | One `CorpusDocument` in the corpus | post, page, record |
| **chunk** | An exact slice of one document, made of whole sentences | split, passage, segment |
| **index** | The persisted `DenseIndex`: chunk vectors plus metadata | vector store, database |
| **fingerprint** | The SHA-256 hash of the corpus content that the index stores | checksum, version hash |
| **embedder** | `HashingEmbedder` or `MiniLMEmbedder` | encoder, vectorizer |
| **query variant** | One text that the rewriter makes from a message | sub-query, paraphrase (for the object) |
| **rewriter** | `RuleRewriter`, `LLMRewriter` or `SingleQuery` | query generator, expander |
| **RRF** | Reciprocal rank fusion of several ranked lists | fusion score, RAG-Fusion |
| **generator** | `ExtractiveGenerator` or `LLMGenerator` | responder, writer |
| **LLM** | A language model behind the `LLM` interface, for example Ollama | AI, model (alone) |
| **citation** | A marker `[n]` that points to source number n of the reply | reference, footnote |
| **session memory** | `SessionMemory`: the bounded, opt-in history of one session | chat history, conversation store |
| **red-team set** | The synthetic messages with a known risk level | test prompts, attack set |
| **template** | One message pattern of the red-team set, with a `template_id` | prompt pattern, seed sentence |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **assess** | Give a risk level and categories to a message |
| **escalate** | Return the escalation message and stop the pipeline |
| **refuse** | Return a scope-limit reply for a diagnosis or medication message |
| **chunk** | Cut a document into chunks of whole sentences |
| **embed** | Change texts into L2-normalised vectors |
| **rewrite** | Make the query variants of a message |
| **retrieve** | Get the top chunks for a message with RRF |
| **fuse** | Combine ranked lists with RRF |
| **generate** | Make the text of an answer from the sources |
| **guard** | Check generated text and remove or block unsafe parts |
| **transcribe** | Change audio bytes into a message |
| **synthesize** | Change reply text into audio bytes |
| **evaluate** | Measure the safety gate, the retrieval or the answers |
