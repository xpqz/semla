# Semantic search

Some hacky AI experiments.

The `faissidx.py` builds a semantic embeddings database in `data`. This takes a while, and uses tokens.

Run the `semla` program to query the index on the command line. You will get back a list of URLs.

Run the webservice:

```
uvicorn semla-ws:app --workers 4
```

or (for prod):

```
gunicorn semla-ws:app -w 4 -k uvicorn.workers.UvicornWorker
```

```
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "how do I serialise and compress an apl array?", "k": 5}'
```

or visit the convenient endpoint http://localhost:8000/docs

The `daal` program uses the custom assistant. It expects the custom assistant id to be available as the environment variable `OPENAI_ASSISTANT_ID`.

Create a custom assistant with `create_assistant.py`. It will upload files to the remote vector store. Note: keeping a remote vector store costs money per month over a certain size. Caveat Emptor.

The `dyaclaude.py` program is an early experiment in making Claude better at APL (not semantic search). A work in progress.

