# Graph Gap Report — qwen2.5-coder-14b-instruct-q4_k_m.gguf

- Run mode: `graph` (analyzed as `graph`)
- Query set: `v1v2`
- Score: **190/200** (95.0%)
- Failures inspected: 10

## Failures by domain

- **cli**: 3 failures
- **general**: 3 failures
- **async**: 2 failures
- **testing**: 1 failure
- **web**: 1 failure

## Most-retrieved categories on failing queries

Categories the graph walk landed on when the query ultimately failed. High counts here suggest either (a) the retrieved examples aren't teaching the right pattern shape, or (b) this category is a fallback when no better match exists.

- `c_basics`: 2
- `js_web`: 2
- `bash_basics`: 2
- `rust_basics`: 2
- `csharp_basics`: 1
- `cli_argparse`: 1
- `data_validation`: 1
- `ruby_basics`: 1
- `js_basics`: 1
- `pattern_context_manager`: 1

## Topic mismatch — possible graph edge gaps

Query domain is X but the top retrieved example category is Y, with no obvious topical overlap. These are prime candidates for **adding edges** to `data/pattern_graph.yaml` or **authoring new category-specific examples**.

- **async** query landed on `c_basics`
  > write an async function that fetches multiple urls concurrently
- **async** query landed on `csharp_basics`
  > how do I run multiple async tasks at the same time?
- **testing** query landed on `c_basics`
  > how do I test async functions with pytest?
- **general** query landed on `js_web`
  > write a function to parse a cron expression
- **cli** query landed on `pattern_context_manager`
  > write a function to safely delete a file if it exists
- **cli** query landed on `rust_basics`
  > write a function that reads the last n lines of a file efficiently
- **general** query landed on `data_structure`
  > make a priority queue class using a heap

## Per-failure details

### async — write an async function that fetches multiple urls concurrently
- failure: missing=`['await']` lines=29 tokens=386
- augmentor: `code_gen` (retrieval=flat)
- retrieved examples:
  - `c_basics` — *Write a C function that implements a dynamic array (vector) with push *

### async — how do I run multiple async tasks at the same time?
- failure: missing=`['gather']` lines=27 tokens=334
- augmentor: `code_gen` (retrieval=flat)
- retrieved examples:
  - `csharp_basics` — *Write a C# interface for an async repository with generic CRUD methods*

### testing — how do I test async functions with pytest?
- failure: missing=`[['def test_', 'async def test', '@pytest.mark.asyncio', 'pytest-asyncio']]` lines=4 tokens=470
- augmentor: `code_gen` (retrieval=flat)
- retrieved examples:
  - `c_basics` — *Write a C function that reads a file into a dynamically allocated stri*

### cli — make a config file loader that reads yaml or json
- failure: missing=`['open']` lines=13 tokens=278
- augmentor: `code_gen` (retrieval=flat)
- retrieved examples:
  - `cli_argparse` — *Config file loader with argparse overrides. Reads a JSON config file, *
  - `data_validation` — *Write a dataclass-based configuration loader that reads a JSON file,
v*
  - `ruby_basics` — *Write a Ruby method that reads a JSON file and returns a hash.*

### general — write a function to parse a cron expression
- failure: missing=`['def ']` lines=78 tokens=737
- augmentor: `code_gen` (retrieval=flat)
- retrieved examples:
  - `js_web` — *Write an Express middleware that validates a JSON body has required fi*

### general — build a function that converts camelCase to snake_case
- failure: missing=`['def ']` lines=6 tokens=43
- augmentor: `code_gen` (retrieval=flat)
- retrieved examples:
  - `js_basics` — *Write a JavaScript function that converts a string to camelCase.*

### cli — write a function to safely delete a file if it exists
- failure: missing=`['def ', 'os']` lines=16 tokens=317
- augmentor: `code_gen` (retrieval=flat)
- retrieved examples:
  - `pattern_context_manager` — *Write an `AtomicFileWrite` context manager that provides safe file wri*
  - `bash_basics` — *Write a bash script that finds and deletes all files older than N days*
  - `rust_basics` — *Write a Rust function that reads a file to a string and returns a Resu*

### cli — write a function that reads the last n lines of a file efficiently
- failure: missing=`['def ']` lines=46 tokens=585
- augmentor: `code_gen` (retrieval=flat)
- retrieved examples:
  - `rust_basics` — *Write a Rust function that reads all lines from a file and returns the*
  - `data_file_io` — *Write a streaming line-by-line file processor using a generator. It ta*
  - `bash_basics` — *Write a bash script that takes a filename argument and counts the numb*

### web — build a simple websocket echo server
- failure: missing=`['def ']` lines=14 tokens=638
- augmentor: `code_gen` (retrieval=flat)
- retrieved examples:
  - `js_web` — *Write a JavaScript class that wraps a WebSocket connection with auto-r*

### general — make a priority queue class using a heap
- failure: missing=`['push']` lines=51 tokens=705
- augmentor: `code_gen` (retrieval=flat)
- retrieved examples:
  - `data_structure` — *Write a `QueueFromStacks` class that implements a FIFO queue using two*
  - `pattern_tree` — *Write a MinHeap class backed by a list. Support insert, extract_min,
p*
  - `java_basics` — *Write a Java generic class for a thread-safe bounded buffer using sync*
