"""
Compatibility package.

Historically, entrypoints imported modules under `src.*`.
The current implementation lives under `main.*`.
This package re-exports the `main` pipeline to keep scripts stable.
"""


