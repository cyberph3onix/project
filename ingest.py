import pathway as pw
from chunking import chunk_text

# Read text once
table = pw.io.fs.read(
    "data/",
    format="plaintext",
    mode="static"
)

# Chunk
chunks = table.select(
    chunk=pw.apply(chunk_text, table.data)
).flatten(pw.this.chunk)

# Write chunks to a file
pw.io.jsonlines.write(
    chunks,
    "chunks.jsonl"
)

pw.run()