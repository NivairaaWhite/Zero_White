# "Like a sister" companion

A ready-made configuration for the PIPPA sister-persona use case.

```bash
zerowhite chat init -w sis
cp examples/persona_sister/config.json sis/config.json
cp -r examples/persona_sister/data/persona sis/data/
zerowhite chat fetch pippa -w sis --limit 20000        # or copy PIPPA's files into sis/data/persona/ yourself
zerowhite chat ingest -w sis                           # prints how many records the persona and safety filters kept/dropped
zerowhite chat train  -w sis --minutes 120
zerowhite chat start  -w sis                           # web UI; /good and /fix teach it what "sister" means to you
```

What it does and does not do:

* Keeps only conversations whose persona text matches the keywords **as whole words** (the original `'sis '` substring trick also matched "analysis").
* Always drops sexual content that involves a minor or a family member (not configurable), and by default all sexually explicit records (`filter.safety.drop_sexual`).
  The detector is a lexicon, so it misses euphemisms: look at your data. See `docs/DATA.md`.
* The system prompt in `config.json` sets the tone; edit it freely. A small model trained on roleplay chats learns the *style* of the data, not a real relationship.
* Mix in general instruction data (`chat fetch alpaca`) so the companion can also answer plain questions.
