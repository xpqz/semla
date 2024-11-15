# Semantic search

Some hacky AI experiments.

You need an environment variable with your OpenAI API key set:
```
export OPENAI_API_KEY="sk-proj-TI...."
```

The `faissidx.py` builds a semantic embeddings database in `data`. This takes a while, and uses tokens.

Run the `semla` program to query the index on the command line. You will get back a list of URLs:

```
% ./semla
Loaded index with 3154 vectors
> how do i serialise and compress an apl array?
https://dyalog.github.io/documentation/20.0/object-reference/miscellaneous/com-data-types (1.0299)
https://dyalog.github.io/documentation/20.0/programming-reference-guide/component-files/introduction (1.0334)
https://dyalog.github.io/documentation/20.0/language-reference-guide/the-i-beam-operator/serialise-array (1.0646)
https://dyalog.github.io/documentation/20.0/programming-reference-guide/introduction/namespaces/serialising-namespaces (1.0819)
https://dyalog.github.io/documentation/20.0/programming-reference-guide/introduction/arrays/arrays (1.1204)

>
```

Run the webservice:

```
uvicorn semla-ws:app --workers 4
```

or (for prod):

```
gunicorn semla-ws:app -w 4 -k uvicorn.workers.UvicornWorker
```

You can test the end point with `curl`:
```
% curl -s -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "how do I serialise and compress an apl array?", "k": 5}' | jq .
{
  "urls": [
    "https://dyalog.github.io/documentation/20.0/object-reference/miscellaneous/com-data-types",
    "https://dyalog.github.io/documentation/20.0/language-reference-guide/the-i-beam-operator/serialise-array",
    "https://dyalog.github.io/documentation/20.0/programming-reference-guide/component-files/introduction",
    "https://dyalog.github.io/documentation/20.0/programming-reference-guide/introduction/namespaces/serialising-namespaces",
    "https://dyalog.github.io/documentation/20.0/programming-reference-guide/introduction/arrays/arrays"
  ],
  "scores": [
    1.032598614692688,
    1.0425920486450195,
    1.058443307876587,
    1.0692082643508911,
    1.1278154850006104
  ]
}
```

or visit the convenient endpoint http://localhost:8000/docs

The `daal` program uses the custom assistant. It expects the custom assistant id to be available as the environment variable `OPENAI_ASSISTANT_ID`.

Create a custom assistant with `create_assistant.py`. It will upload files to the remote vector store. Note: keeping a remote vector store costs money per month over a certain size. Caveat Emptor.

The `dyaclaude.py` program is an early experiment in making Claude better at APL (not semantic search). A work in progress.

