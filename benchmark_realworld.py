#!/usr/bin/env python3
"""
Real-World Query Benchmark — 100 prompts people actually ask coding assistants.

Tests whether the augmentor system produces working code for natural, varied prompts
across all domains. Each query has multiple phrasings to test keyword robustness.

Usage:
    python benchmark_realworld.py                    # Run all, default model
    python benchmark_realworld.py --quick             # 20 queries only
    python benchmark_realworld.py --model models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
    python benchmark_realworld.py --check-routing     # Just check which examples get injected (no model needed)
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class RealWorldQuery:
    """A realistic user query with expected code markers."""
    query: str
    domain: str  # web, database, async, testing, cli, algorithm, pattern, data, general
    must_contain: list[str] = field(default_factory=list)  # strings that MUST appear in output
    must_not_contain: list[str] = field(default_factory=list)  # strings that must NOT appear
    min_lines: int = 3  # minimum code lines expected


def build_realworld_queries() -> list[RealWorldQuery]:
    """100 queries real users would ask, with varied phrasing."""
    return [
        # ── Algorithm / Math (10) ──
        RealWorldQuery("write me a fibonacci function", "algorithm",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("i need code to sort a list of numbers without using the built-in sort", "algorithm",
                       must_contain=["def "], min_lines=5),
        RealWorldQuery("can you make a binary search function?", "algorithm",
                       must_contain=["def ", "mid"], min_lines=5),
        RealWorldQuery("build a calculator that shows pi to n decimal places", "algorithm",
                       must_contain=["Decimal"], must_not_contain=["math.pi *"]),
        RealWorldQuery("compute e to 50 decimal places", "algorithm",
                       must_contain=["Decimal", "getcontext"], must_not_contain=["math.e"]),
        RealWorldQuery("write a function to check if a number is prime", "algorithm",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("give me code to reverse a linked list", "algorithm",
                       must_contain=["class ", "next"], min_lines=5),
        RealWorldQuery("how do I find all permutations of a string?", "algorithm",
                       must_contain=["def "], min_lines=4),
        RealWorldQuery("write a function that computes square root of 2 to 100 digits", "algorithm",
                       must_contain=["Decimal"], must_not_contain=["math.sqrt"]),
        RealWorldQuery("implement merge sort in python", "algorithm",
                       must_contain=["def ", "merge"], min_lines=8),

        # ── Data Structures (8) ──
        RealWorldQuery("build an LRU cache class with get and put", "pattern",
                       must_contain=["class ", "def get", "def put"], min_lines=10),
        RealWorldQuery("make a stack class with push pop and peek", "pattern",
                       must_contain=["class ", "push", "pop"], min_lines=8),
        RealWorldQuery("write a binary search tree with insert and search", "pattern",
                       must_contain=["class ", "insert"], min_lines=10),
        RealWorldQuery("create a queue using two stacks", "pattern",
                       must_contain=["class "], min_lines=8),
        RealWorldQuery("implement a trie data structure for word lookup", "pattern",
                       must_contain=["class ", "insert"], min_lines=10),
        RealWorldQuery("write a hash map from scratch without using dict", "pattern",
                       must_contain=["class ", "def "], min_lines=10),
        RealWorldQuery("make a min heap class with push and pop", "pattern",
                       must_contain=["class ", "push", "pop"], min_lines=8),
        RealWorldQuery("build a graph class with add_edge and bfs", "pattern",
                       must_contain=["class ", "bfs"], min_lines=8),

        # ── Web / API (10) ──
        RealWorldQuery("write a fastapi endpoint that accepts json and returns a response", "web",
                       must_contain=["fastapi", "def "], min_lines=5),
        RealWorldQuery("create a REST API with get post and delete routes", "web",
                       must_contain=["def "], min_lines=8),
        RealWorldQuery("how do I add authentication middleware to my api?", "web",
                       must_contain=["def "], min_lines=5),
        RealWorldQuery("write a pydantic model for validating user registration", "web",
                       must_contain=["class ", "BaseModel"], min_lines=4),
        RealWorldQuery("build a simple http server that serves json", "web",
                       must_contain=["def "], min_lines=5),
        RealWorldQuery("make an endpoint that uploads a file", "web",
                       must_contain=["def "], min_lines=4),
        RealWorldQuery("write middleware that logs how long each request takes", "web",
                       must_contain=["def ", "time"], min_lines=5),
        RealWorldQuery("create a rate limiter for my api endpoints", "web",
                       must_contain=["def "], min_lines=5),
        RealWorldQuery("how do I handle CORS in a python web server?", "web",
                       must_contain=["def "], min_lines=3),
        RealWorldQuery("write an api endpoint with query parameters and path params", "web",
                       must_contain=["def "], min_lines=4),

        # ── Database (10) ──
        RealWorldQuery("write a sqlite database class with create read update delete", "database",
                       must_contain=["sqlite3", "def "], min_lines=10),
        RealWorldQuery("how do I connect to a sqlite database and run a query?", "database",
                       must_contain=["sqlite3", "connect"], min_lines=3),
        RealWorldQuery("create a function that inserts multiple rows into sqlite", "database",
                       must_contain=["sqlite3", "executemany"], min_lines=4),
        RealWorldQuery("write a context manager for database connections that auto commits", "database",
                       must_contain=["def ", "commit"], min_lines=5),
        RealWorldQuery("build a simple migration system for sqlite", "database",
                       must_contain=["def ", "sqlite3"], min_lines=8),
        RealWorldQuery("write a repository pattern class for managing users in a database", "database",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("how to do parameterized queries to prevent sql injection?", "database",
                       must_contain=["?"], min_lines=3),
        RealWorldQuery("create a connection pool for sqlite", "database",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("write code to export a sqlite table to csv", "database",
                       must_contain=["csv", "sqlite3"], min_lines=5),
        RealWorldQuery("build a key-value store backed by sqlite", "database",
                       must_contain=["sqlite3", "def get", "def set"], min_lines=8),

        # ── Async (8) ──
        RealWorldQuery("write an async function that fetches multiple urls concurrently", "async",
                       must_contain=["async ", "await"], min_lines=5),
        RealWorldQuery("create an async producer consumer with a queue", "async",
                       must_contain=["async ", "Queue"], min_lines=8),
        RealWorldQuery("how do I run multiple async tasks at the same time?", "async",
                       must_contain=["async ", "gather"], min_lines=4),
        RealWorldQuery("write an async rate limiter using semaphore", "async",
                       must_contain=["async ", "Semaphore"], min_lines=5),
        RealWorldQuery("build an async context manager for timing operations", "async",
                       must_contain=["async ", "__aenter__"], min_lines=5),
        RealWorldQuery("make a worker pool that processes items from an async queue", "async",
                       must_contain=["async ", "Queue"], min_lines=8),
        RealWorldQuery("write an async generator that yields numbers with a delay", "async",
                       must_contain=["async def", "yield"], min_lines=4),
        RealWorldQuery("how to use asyncio.wait with a timeout?", "async",
                       must_contain=["async", "wait"], min_lines=4),

        # ── Testing (8) ──
        RealWorldQuery("write a pytest fixture that sets up a temporary database", "testing",
                       must_contain=["@pytest.fixture", "yield"], min_lines=5),
        RealWorldQuery("how do I mock an api call in my tests?", "testing",
                       must_contain=["mock", "patch"], min_lines=5),
        RealWorldQuery("write parametrized tests for a function with multiple inputs", "testing",
                       must_contain=["parametrize"], min_lines=4),
        RealWorldQuery("create a test factory for generating user objects", "testing",
                       must_contain=["def make_", "return"], min_lines=5),
        RealWorldQuery("how to mock file reading in a unit test?", "testing",
                       must_contain=["mock_open"], min_lines=4),
        RealWorldQuery("write tests for a calculator class with add subtract multiply", "testing",
                       must_contain=["def test_", "assert"], min_lines=5),
        RealWorldQuery("create a custom assertion helper for checking api responses", "testing",
                       must_contain=["def assert_", "raise"], min_lines=4),
        RealWorldQuery("how do I test async functions with pytest?", "testing",
                       must_contain=["async", "test"], min_lines=4),

        # ── CLI / Scripting (8) ──
        RealWorldQuery("build a cli tool with subcommands using argparse", "cli",
                       must_contain=["argparse", "add_subparsers"], min_lines=8),
        RealWorldQuery("write a python script with proper logging setup", "cli",
                       must_contain=["logging", "handler"], min_lines=5),
        RealWorldQuery("how do I run a shell command and capture the output?", "cli",
                       must_contain=["subprocess"], min_lines=3),
        RealWorldQuery("create a script that watches a directory for new files", "cli",
                       must_contain=["def "], min_lines=5),
        RealWorldQuery("write a progress bar for a long running task", "cli",
                       must_contain=["def "], min_lines=4),
        RealWorldQuery("make a config file loader that reads yaml or json", "cli",
                       must_contain=["def ", "open"], min_lines=5),
        RealWorldQuery("write a script that processes command line arguments", "cli",
                       must_contain=["argparse"], min_lines=4),
        RealWorldQuery("build a rotating log file system", "cli",
                       must_contain=["RotatingFileHandler"], min_lines=4),

        # ── Data Processing (8) ──
        RealWorldQuery("write a function to read a csv file into a list of dicts", "data",
                       must_contain=["csv", "DictReader"], min_lines=3),
        RealWorldQuery("build a data pipeline with map filter and reduce", "data",
                       must_contain=["def ", "map", "filter"], min_lines=5),
        RealWorldQuery("flatten a nested dictionary to dot notation keys", "data",
                       must_contain=["def flatten"], min_lines=5),
        RealWorldQuery("write a json schema validator", "data",
                       must_contain=["def ", "validate"], min_lines=5),
        RealWorldQuery("create a function to process a large file line by line", "data",
                       must_contain=["def ", "open"], min_lines=4),
        RealWorldQuery("write code to merge two sorted lists", "data",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("build a dataclass config loader from a json file", "data",
                       must_contain=["dataclass", "json"], min_lines=5),
        RealWorldQuery("write a function that groups a list of dicts by a key", "data",
                       must_contain=["def ", "return"], min_lines=4),

        # ── Patterns (10) ──
        RealWorldQuery("write a decorator that retries a function 3 times on failure", "pattern",
                       must_contain=["def ", "wrapper", "retry"], min_lines=6),
        RealWorldQuery("create a state machine class with transitions", "pattern",
                       must_contain=["class ", "transition"], min_lines=8),
        RealWorldQuery("build an event emitter with on emit and off methods", "pattern",
                       must_contain=["class ", "on", "emit"], min_lines=8),
        RealWorldQuery("write a singleton pattern in python", "pattern",
                       must_contain=["class "], min_lines=4),
        RealWorldQuery("create an iterator class that generates fibonacci numbers", "pattern",
                       must_contain=["class ", "__iter__"], min_lines=6),
        RealWorldQuery("write a context manager for temporarily changing a directory", "pattern",
                       must_contain=["def ", "__enter__"], min_lines=5),
        RealWorldQuery("implement the observer pattern in python", "pattern",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("build a simple template engine that replaces {{variables}}", "pattern",
                       must_contain=["def ", "{{"], min_lines=4),
        RealWorldQuery("write a middleware chain that processes requests in order", "pattern",
                       must_contain=["def ", "next"], min_lines=5),
        RealWorldQuery("create a rate limiter using the token bucket algorithm", "pattern",
                       must_contain=["class ", "def "], min_lines=6),

        # ── General / Mixed (20) ──
        RealWorldQuery("write a function that validates an email address", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("create a password hasher with salt", "general",
                       must_contain=["def ", "hash"], min_lines=4),
        RealWorldQuery("write a function to convert between celsius and fahrenheit", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("build a simple calculator that handles + - * /", "general",
                       must_contain=["def "], min_lines=4),
        RealWorldQuery("write code to download a file from a url", "general",
                       must_contain=["def ", "http"], min_lines=4),
        RealWorldQuery("create a function that generates a random password", "general",
                       must_contain=["def ", "random"], min_lines=4),
        RealWorldQuery("write a class that represents a playing card deck with shuffle and deal", "general",
                       must_contain=["class ", "shuffle", "deal"], min_lines=8),
        RealWorldQuery("build a simple todo list class with add remove and list methods", "general",
                       must_contain=["class ", "add", "remove"], min_lines=8),
        RealWorldQuery("write a function to find duplicate files in a directory", "general",
                       must_contain=["def ", "os"], min_lines=5),
        RealWorldQuery("create a countdown timer that prints remaining time", "general",
                       must_contain=["def ", "time"], min_lines=4),
        RealWorldQuery("write a function to prettify json with indentation", "general",
                       must_contain=["json", "indent"], min_lines=3),
        RealWorldQuery("make a class that reads and writes ini config files", "general",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("write a function to chunk a list into groups of n", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("create a simple encryption function using xor", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("write a url shortener using a hash", "general",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("build a class that manages a shopping cart", "general",
                       must_contain=["class ", "add"], min_lines=8),
        RealWorldQuery("write a function that converts markdown headers to html", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("create a simple spell checker using edit distance", "general",
                       must_contain=["def "], min_lines=5),
        RealWorldQuery("write a function to parse a cron expression", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("build a retry decorator with exponential backoff", "general",
                       must_contain=["def ", "wrapper", "sleep"], min_lines=6),
    ]


def build_realworld_queries_v2() -> list[RealWorldQuery]:
    """100 MORE queries — project-oriented, messier phrasing, edge cases."""
    return [
        # ── String / Text Manipulation (10) ──
        RealWorldQuery("write a function that counts how many times each word appears in a string", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("strip all html tags from a string", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("write a slug generator that converts titles to url-friendly strings", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("how do i check if a string is a valid ip address", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("make a function that wraps text to a max line width", "general",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("write a caesar cipher encrypt and decrypt", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("build a function that converts camelCase to snake_case", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("create a string diff function that shows what changed between two strings", "general",
                       must_contain=["def "], min_lines=4),
        RealWorldQuery("write a function to extract all urls from a block of text", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("build a simple tokenizer that splits code into tokens", "general",
                       must_contain=["def ", "return"], min_lines=5),

        # ── File System / OS (10) ──
        RealWorldQuery("write a function that recursively finds all python files in a directory", "cli",
                       must_contain=["def ", ".py"], min_lines=3),
        RealWorldQuery("create a file watcher that detects when files change", "cli",
                       must_contain=["def "], min_lines=5),
        RealWorldQuery("write a script to rename all files in a folder to lowercase", "cli",
                       must_contain=["os", "rename"], min_lines=4),
        RealWorldQuery("build a function that calculates the total size of a directory", "cli",
                       must_contain=["def ", "size"], min_lines=4),
        RealWorldQuery("how do i create a temp file write to it and read it back", "cli",
                       must_contain=["tempfile", "write"], min_lines=4),
        RealWorldQuery("write a function to safely delete a file if it exists", "cli",
                       must_contain=["def ", "os"], min_lines=3),
        RealWorldQuery("make a zip archive of a folder", "cli",
                       must_contain=["zip", "def "], min_lines=4),
        RealWorldQuery("write a function that reads the last n lines of a file efficiently", "cli",
                       must_contain=["def ", "open"], min_lines=4),
        RealWorldQuery("create a lockfile mechanism to prevent running a script twice", "cli",
                       must_contain=["def ", "lock"], min_lines=5),
        RealWorldQuery("build a function that copies a directory tree with filtering", "cli",
                       must_contain=["def ", "os"], min_lines=4),

        # ── Networking / HTTP (10) ──
        RealWorldQuery("write a function to download a file from a url with a progress callback", "web",
                       must_contain=["def ", "url"], min_lines=5),
        RealWorldQuery("make a simple http client that does get and post requests", "web",
                       must_contain=["def ", "request"], min_lines=5),
        RealWorldQuery("write a function that checks if a website is up or down", "web",
                       must_contain=["def ", "http"], min_lines=4),
        RealWorldQuery("build a simple websocket echo server", "web",
                       must_contain=["def "], min_lines=5),
        RealWorldQuery("create a function that parses url query parameters into a dict", "web",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("write a dns lookup function that returns all ip addresses for a hostname", "web",
                       must_contain=["def ", "socket"], min_lines=3),
        RealWorldQuery("build a retry wrapper for http requests with exponential backoff", "web",
                       must_contain=["def ", "retry"], min_lines=6),
        RealWorldQuery("write a function that sends a json post request with headers", "web",
                       must_contain=["def ", "json"], min_lines=4),
        RealWorldQuery("create a simple tcp server that echoes back whatever it receives", "web",
                       must_contain=["socket", "bind"], min_lines=6),
        RealWorldQuery("write an api client class with get post put delete methods", "web",
                       must_contain=["class ", "def get"], min_lines=8),

        # ── Classes / OOP (10) ──
        RealWorldQuery("write a bank account class with deposit withdraw and transfer", "general",
                       must_contain=["class ", "deposit", "withdraw"], min_lines=8),
        RealWorldQuery("create a linked list class with append prepend and delete", "general",
                       must_contain=["class ", "append"], min_lines=8),
        RealWorldQuery("build a matrix class that supports addition and multiplication", "general",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("write a class that represents a deck of cards with shuffle and draw", "general",
                       must_contain=["class ", "shuffle"], min_lines=8),
        RealWorldQuery("make a priority queue class using a heap", "general",
                       must_contain=["class ", "push", "pop"], min_lines=6),
        RealWorldQuery("create a class that wraps a list and adds statistics methods like mean and median", "general",
                       must_contain=["class ", "mean"], min_lines=6),
        RealWorldQuery("write a money class that handles dollars and cents without floating point errors", "general",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("build a simple orm-style model class that maps to a dict", "general",
                       must_contain=["class ", "def "], min_lines=6),
        RealWorldQuery("create an immutable point class with x and y coordinates", "general",
                       must_contain=["class ", "x", "y"], min_lines=4),
        RealWorldQuery("write a tree node class with methods to add children and find by value", "general",
                       must_contain=["class ", "add", "find"], min_lines=6),

        # ── Error Handling / Validation (8) ──
        RealWorldQuery("write a custom exception hierarchy for a web application", "general",
                       must_contain=["class ", "Exception"], min_lines=5),
        RealWorldQuery("create a function that validates a credit card number with luhn algorithm", "algorithm",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("write a retry function that catches specific exception types", "pattern",
                       must_contain=["def ", "except"], min_lines=5),
        RealWorldQuery("build an input validator that checks types ranges and required fields", "data",
                       must_contain=["def ", "validate"], min_lines=5),
        RealWorldQuery("write a function that gracefully handles json decode errors", "general",
                       must_contain=["def ", "json", "except"], min_lines=4),
        RealWorldQuery("create a context manager that suppresses and logs specific exceptions", "pattern",
                       must_contain=["def ", "__enter__"], min_lines=5),
        RealWorldQuery("write a function that validates a phone number format", "general",
                       must_contain=["def ", "return"], min_lines=3),
        RealWorldQuery("build an error collector that gathers multiple validation errors before raising", "general",
                       must_contain=["def ", "error"], min_lines=5),

        # ── Concurrency / Parallel (8) ──
        RealWorldQuery("write a thread pool that processes a list of tasks", "pattern",
                       must_contain=["Thread", "def "], min_lines=5),
        RealWorldQuery("create a producer consumer pattern using threading and a queue", "pattern",
                       must_contain=["Thread", "Queue"], min_lines=8),
        RealWorldQuery("write a function that runs multiple functions in parallel and collects results", "pattern",
                       must_contain=["def ", "result"], min_lines=5),
        RealWorldQuery("build a simple job scheduler that runs tasks at intervals", "pattern",
                       must_contain=["def ", "time"], min_lines=6),
        RealWorldQuery("write a deadlock-free transfer between two bank accounts using locks", "pattern",
                       must_contain=["Lock", "def "], min_lines=6),
        RealWorldQuery("create a read-write lock that allows multiple readers but single writer", "pattern",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("write an async web scraper that fetches 10 pages concurrently", "async",
                       must_contain=["async ", "await"], min_lines=6),
        RealWorldQuery("build a background task runner that queues and executes functions", "pattern",
                       must_contain=["class ", "def "], min_lines=6),

        # ── Data Formats / Serialization (8) ──
        RealWorldQuery("write a function that converts xml to a python dict", "data",
                       must_contain=["def ", "xml"], min_lines=5),
        RealWorldQuery("create a csv to json converter", "data",
                       must_contain=["def ", "csv", "json"], min_lines=4),
        RealWorldQuery("write a yaml config file parser with defaults", "data",
                       must_contain=["def ", "yaml"], min_lines=4),
        RealWorldQuery("build a function that pretty prints any nested data structure", "data",
                       must_contain=["def ", "print"], min_lines=4),
        RealWorldQuery("write a serializer that converts python objects to json including dates", "data",
                       must_contain=["def ", "date"], min_lines=4),
        RealWorldQuery("create a function that deep merges two dictionaries", "data",
                       must_contain=["def ", "merge"], min_lines=5),
        RealWorldQuery("write a function that converts a flat list of records into a tree structure", "data",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("build a toml parser that reads key value pairs from a string", "data",
                       must_contain=["def ", "return"], min_lines=5),

        # ── Database Advanced (6) ──
        RealWorldQuery("write a full text search function using sqlite fts5", "database",
                       must_contain=["sqlite3", "fts"], min_lines=5),
        RealWorldQuery("create a caching layer that stores query results in sqlite with expiry", "database",
                       must_contain=["sqlite3", "def "], min_lines=8),
        RealWorldQuery("build a simple event store that saves events with timestamps in sqlite", "database",
                       must_contain=["sqlite3", "def "], min_lines=6),
        RealWorldQuery("write a function to backup a sqlite database to a file", "database",
                       must_contain=["sqlite3", "backup"], min_lines=4),
        RealWorldQuery("create an audit log table that tracks all changes to a users table", "database",
                       must_contain=["class ", "log"], min_lines=5),
        RealWorldQuery("write a function that exports sqlite query results to a pandas dataframe", "database",
                       must_contain=["sqlite3", "pandas"], min_lines=4),

        # ── DevOps / Automation (6) ──
        RealWorldQuery("write a health check function that pings a list of services", "cli",
                       must_contain=["def ", "health"], min_lines=5),
        RealWorldQuery("create a simple cron-like scheduler in python", "cli",
                       must_contain=["def ", "schedule"], min_lines=6),
        RealWorldQuery("write a script that monitors cpu and memory usage", "cli",
                       must_contain=["def "], min_lines=4),
        RealWorldQuery("build a deployment script that runs commands over ssh", "cli",
                       must_contain=["def ", "ssh"], min_lines=5),
        RealWorldQuery("write a function that generates a requirements.txt from import statements", "cli",
                       must_contain=["def ", "import"], min_lines=5),
        RealWorldQuery("create a log parser that extracts error lines and counts by type", "cli",
                       must_contain=["def ", "error"], min_lines=5),

        # ── Math / Science (6) ──
        RealWorldQuery("write a function that solves a system of linear equations", "algorithm",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("create a statistics class with mean median mode and standard deviation", "algorithm",
                       must_contain=["class ", "mean"], min_lines=8),
        RealWorldQuery("write a function that generates the collatz sequence for a number", "algorithm",
                       must_contain=["def ", "return"], min_lines=4),
        RealWorldQuery("build a simple neural network perceptron from scratch", "algorithm",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("write a function that computes the greatest common divisor of a list of numbers", "algorithm",
                       must_contain=["def ", "gcd"], min_lines=3),
        RealWorldQuery("create a polynomial class that supports addition and evaluation", "algorithm",
                       must_contain=["class ", "def "], min_lines=6),

        # ── Security / Crypto (4) ──
        RealWorldQuery("write a password strength checker that scores passwords", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("create a function that generates and verifies hmac signatures", "general",
                       must_contain=["hmac", "def "], min_lines=4),
        RealWorldQuery("write a jwt token encoder and decoder without external libraries", "general",
                       must_contain=["def ", "encode"], min_lines=6),
        RealWorldQuery("build a simple hash table with chaining for collision resolution", "general",
                       must_contain=["class ", "def "], min_lines=8),

        # ── Mini Projects (14) ──
        RealWorldQuery("build a command line tic tac toe game", "general",
                       must_contain=["def ", "board"], min_lines=10),
        RealWorldQuery("write a markdown to html converter for headers paragraphs and bold", "general",
                       must_contain=["def ", "return"], min_lines=6),
        RealWorldQuery("create a simple chat server using sockets", "web",
                       must_contain=["socket", "def "], min_lines=8),
        RealWorldQuery("build a url shortener with a dictionary backend", "general",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("write a simple regex engine that handles . * and literal matches", "algorithm",
                       must_contain=["def ", "match"], min_lines=6),
        RealWorldQuery("create a task queue with priorities and worker threads", "pattern",
                       must_contain=["class ", "Queue"], min_lines=8),
        RealWorldQuery("build a simple version of git diff that compares two strings line by line", "general",
                       must_contain=["def ", "diff"], min_lines=5),
        RealWorldQuery("write a json path query function like jq", "data",
                       must_contain=["def ", "return"], min_lines=5),
        RealWorldQuery("create a simple pub sub message broker", "pattern",
                       must_contain=["class ", "publish", "subscribe"], min_lines=8),
        RealWorldQuery("build a rate limited api client that respects retry-after headers", "web",
                       must_contain=["class ", "def "], min_lines=8),
        RealWorldQuery("write a simple load balancer that distributes requests round robin", "web",
                       must_contain=["class ", "def "], min_lines=5),
        RealWorldQuery("create a plugin system that discovers and loads python modules from a directory", "pattern",
                       must_contain=["def ", "import"], min_lines=5),
        RealWorldQuery("build a simple key value store with get set delete and ttl expiry", "general",
                       must_contain=["class ", "def get"], min_lines=8),
        RealWorldQuery("write a function that generates a pdf invoice given order data", "general",
                       must_contain=["def "], min_lines=5),
    ]


def extract_code(response: str) -> str:
    """Extract code from markdown code blocks."""
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()
    if "```" in response:
        parts = response.split("```")
        if len(parts) > 1:
            code = parts[1]
            if code.startswith("\n"):
                code = code[1:]
            return code.split("```")[0].strip()
    return response.strip()


def check_query(code: str, query: RealWorldQuery) -> dict:
    """Check if generated code meets the query requirements."""
    code_lower = code.lower()
    lines = [l for l in code.strip().split("\n") if l.strip() and not l.strip().startswith("#")]

    results = {
        "has_code": len(lines) >= query.min_lines,
        "min_lines": len(lines) >= query.min_lines,
        "line_count": len(lines),
        "must_contain_pass": [],
        "must_contain_fail": [],
        "must_not_contain_pass": [],
        "must_not_contain_fail": [],
    }

    for marker in query.must_contain:
        if marker.lower() in code_lower:
            results["must_contain_pass"].append(marker)
        else:
            results["must_contain_fail"].append(marker)

    for marker in query.must_not_contain:
        if marker.lower() in code_lower:
            results["must_not_contain_fail"].append(marker)
        else:
            results["must_not_contain_pass"].append(marker)

    total_checks = len(query.must_contain) + len(query.must_not_contain) + 1  # +1 for min_lines
    passed = (len(results["must_contain_pass"]) + len(results["must_not_contain_pass"])
              + (1 if results["min_lines"] else 0))
    results["score"] = passed / total_checks if total_checks > 0 else 1.0
    results["passed"] = len(results["must_contain_fail"]) == 0 and len(results["must_not_contain_fail"]) == 0 and results["min_lines"]

    return results


def check_routing_only(queries: list[RealWorldQuery]):
    """Check which examples get injected for each query without running the model."""
    from engine.augmentors import AugmentorRouter, FAILURE_PATTERNS
    from engine.embedder import get_embedder

    router = AugmentorRouter(yaml_dir="data/augmentor_examples")
    embedder = get_embedder()
    router.init_embeddings(embedder)
    router.use_auto_augmentors(469)  # Simulate 0.5B

    domains = {}
    routed = 0
    unrouted = 0

    for q in queries:
        aug = router.select_augmentor(q.query, "code_gen")
        if not aug:
            continue

        forced = aug._check_failure_patterns(q.query)
        examples = aug._retrieve_for_mode(q.query)
        route_type = "FORCED" if forced else "SIMILARITY"
        cat = examples[0].category if examples else "NONE"

        if cat == "NONE" or (not forced and examples and examples[0].category == "basic"):
            unrouted += 1
            status = "MISS"
        else:
            routed += 1
            status = "HIT"

        if q.domain not in domains:
            domains[q.domain] = {"hit": 0, "miss": 0}
        domains[q.domain]["hit" if status == "HIT" else "miss"] += 1

        print(f"  {status:4s} {route_type:10s} {cat:30s} <- {q.query[:60]}")

    print(f"\n  Routing Coverage: {routed}/{routed + unrouted} ({routed / (routed + unrouted) * 100:.0f}%)")
    print(f"\n  Per domain:")
    for domain in sorted(domains):
        d = domains[domain]
        total = d["hit"] + d["miss"]
        print(f"    {domain:12s} {d['hit']}/{total} ({d['hit'] / total * 100:.0f}%)")


def run_benchmark(model_path: Path, queries: list[RealWorldQuery], gpu_layers: int = 99,
                  threads: int = 8, context_length: int = 4096) -> dict:
    """Run all queries through the model and check results."""
    from llama_cpp import Llama
    from engine.augmentors import AugmentorRouter
    from engine.embedder import get_embedder
    from benchmark_exec import detect_chat_format

    # Load model
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    chat_format = detect_chat_format(str(model_path))
    print(f"\n  Model: {model_path.stem} ({model_size_mb:.0f} MB)")
    print(f"  Format: {chat_format}")
    print(f"  Queries: {len(queries)}")

    model = Llama(
        model_path=str(model_path), n_ctx=context_length,
        n_gpu_layers=gpu_layers, n_threads=threads, n_batch=512, verbose=False,
    )

    # Setup augmentors
    router = AugmentorRouter(yaml_dir="data/augmentor_examples")
    embedder = get_embedder()
    router.init_embeddings(embedder)
    router.use_auto_augmentors(model_size_mb)

    results = []
    stop = ["</s>", "<|im_end|>", "<|end|>", "<|eot_id|>"]

    for i, q in enumerate(queries):
        print(f"  [{i + 1}/{len(queries)}] {q.query[:60]}...", end=" ", flush=True)

        # Get augmentor prompt
        aug = router.select_augmentor(q.query, "code_gen")
        if aug:
            prompt = aug.build_prompt(q.query, chat_format)
        else:
            from benchmark_exec import wrap_chat
            system = "You are a Python coding assistant. Write clean, correct, complete Python code in ```python blocks."
            prompt = wrap_chat(system, q.query, chat_format)

        start = time.monotonic()
        output = model(prompt, max_tokens=512, temperature=0.2, stop=stop, echo=False)
        elapsed = time.monotonic() - start

        response = output["choices"][0]["text"].strip()
        code = extract_code(response)
        checks = check_query(code, q)

        status = "PASS" if checks["passed"] else "FAIL"
        print(f"{status} ({checks['score']:.0%}) [{elapsed:.1f}s]")

        if not checks["passed"]:
            if checks["must_contain_fail"]:
                print(f"         missing: {checks['must_contain_fail']}")
            if checks["must_not_contain_fail"]:
                print(f"         unwanted: {checks['must_not_contain_fail']}")
            if not checks["min_lines"]:
                print(f"         too short: {checks['line_count']} lines (need {q.min_lines})")

        results.append({
            "query": q.query,
            "domain": q.domain,
            "passed": checks["passed"],
            "score": checks["score"],
            "time": round(elapsed, 2),
            "lines": checks["line_count"],
            "missing": checks["must_contain_fail"],
            "unwanted": checks["must_not_contain_fail"],
        })

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / total if total else 0

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed}/{total} passed ({passed / total * 100:.0f}%)")
    print(f"  Average score: {avg_score:.0%}")

    # Per domain
    domains = {}
    for r in results:
        d = r["domain"]
        if d not in domains:
            domains[d] = {"passed": 0, "total": 0, "score_sum": 0}
        domains[d]["total"] += 1
        domains[d]["score_sum"] += r["score"]
        if r["passed"]:
            domains[d]["passed"] += 1

    print(f"\n  Per domain:")
    for domain in sorted(domains):
        d = domains[domain]
        print(f"    {domain:12s} {d['passed']}/{d['total']} ({d['passed'] / d['total'] * 100:.0f}%)"
              f"  avg={d['score_sum'] / d['total']:.0%}")

    return {"total": total, "passed": passed, "avg_score": avg_score, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Real-World Query Benchmark")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--set", type=int, choices=[1, 2], default=1,
                        help="Query set: 1=original 100, 2=project-oriented 100")
    parser.add_argument("--both", action="store_true", help="Run both sets (200 queries)")
    parser.add_argument("--quick", action="store_true", help="Run 20 queries only")
    parser.add_argument("--check-routing", action="store_true", help="Check routing only (no model)")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=str, default="benchmark_realworld_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.both:
        queries = build_realworld_queries() + build_realworld_queries_v2()
        print(f"  Running both sets: {len(queries)} queries")
    elif args.set == 2:
        queries = build_realworld_queries_v2()
        print(f"  Set 2 (project-oriented): {len(queries)} queries")
    else:
        queries = build_realworld_queries()
        print(f"  Set 1 (fundamentals): {len(queries)} queries")

    if args.quick:
        # Take 2 from each domain for variety
        by_domain = {}
        for q in queries:
            by_domain.setdefault(q.domain, []).append(q)
        queries = []
        for domain_queries in by_domain.values():
            queries.extend(domain_queries[:2])
        print(f"  Quick mode: {len(queries)} queries")

    if args.check_routing:
        print(f"\n  Routing Check ({len(queries)} queries)")
        print(f"  {'=' * 60}")
        check_routing_only(queries)
        return

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        models = sorted(Path("models").glob("*.gguf"), key=lambda p: p.stat().st_size)
        if not models:
            print("No models found. Use --model or place .gguf files in models/")
            sys.exit(1)
        model_path = models[0]  # smallest by default

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    data = run_benchmark(model_path, queries, args.gpu_layers, args.threads)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
