#!/usr/bin/env python

"""
See https://platform.openai.com/docs/assistants/tools/file-search/quickstart
"""

import itertools
import os
from typing_extensions import override
from openai import AssistantEventHandler, OpenAI

client = OpenAI()

def chunked(iterable, chunk_size):
    """
    Yield successive chunks from the iterator of a specified size.
    """
    if chunk_size < 1:
        raise ValueError('chunk_size must be at least one')
    it = iter(iterable)
    while batch := list(itertools.islice(it, chunk_size)):
        yield batch


def list_markdown_files(directories):
    markdown_files = []
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.md'):
                    full_path = os.path.join(root, file)
                    markdown_files.append(os.path.abspath(full_path))
    return markdown_files


def make_vector_store(directories, store_name):
    """
    Ready the files for upload to OpenAI

    Use the upload and poll SDK helper to upload the files, add them to the vector store,
    and poll the status of the file batch for completion.
    """
    vector_store = client.beta.vector_stores.create(name=store_name)
    file_streams = [open(path, "rb") for path in list_markdown_files(directories)]
    batches = []

    for ch in chunked(file_streams, 100):
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=ch
        )
        batches.append(file_batch)

    print(f'\nVector store created')
    return vector_store

def create_assistant(vs, instructions, model):
    return client.beta.assistants.create(
        instructions=instructions,
        model=model,
        tools=[{"type": "file_search"}],
    )

class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    @override
    def on_message_done(self, message) -> None:
        # print a citation to the file searched
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f"[{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")

        print(message_content.value)
        print("\n".join(citations))

if __name__ == "__main__":

    instructions = 'You are a helpful assistant, expert in Dyalog APL. Use your knowledge base to answer questions about Dyalog APL. Use ⎕IO←1 for all examples.'
    model = 'gpt-4o'
    assistant = client.beta.assistants.create(
        name="Dyalog APL AI Librarian (DAAL)",
        instructions=instructions,
        model=model,
        tools=[{"type": "file_search"}],
    )
    
    vector_store = make_vector_store(['../documentation'], 'dyalog-docs')
    
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
