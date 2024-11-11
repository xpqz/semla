# Semantic search

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

The `daal` program uses the custom assistant.

