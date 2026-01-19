#!/usr/bin/env python3
"""
MCP Server for RLM (Recursive Language Models).

Exposes RLM capabilities as MCP tools for Claude Desktop and other MCP clients.
This allows processing of large contexts that exceed normal LLM context windows.

Installation:
    cd /path/to/rlm
    pip install -e .              # Install the main rlm package
    pip install -e ./mcp          # Install this MCP server

Usage:
    rlm-mcp                       # Run as stdio server (for Claude Desktop)

Based on: https://github.com/modelcontextprotocol/python-sdk
"""

import glob
import os
import sys

from mcp.server.fastmcp import FastMCP

# Add parent directory to path so we can import rlm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm import RLM

# Configuration from environment
DEFAULT_MODEL = os.environ.get("RLM_MODEL", "claude-opus-4-5-20251101")
DEFAULT_SUB_MODEL = os.environ.get("RLM_SUB_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Initialize MCP server
mcp = FastMCP("rlm-server")


def _create_rlm(
    model: str = DEFAULT_MODEL,
    max_depth: int = 1,
    verbose: bool = False,
) -> RLM:
    """Create an RLM instance with the specified configuration."""
    return RLM(
        backend="anthropic",
        backend_kwargs={
            "model_name": model,
            "api_key": ANTHROPIC_API_KEY,
        },
        environment="local",
        max_depth=max_depth,
        max_iterations=30,
        verbose=verbose,
    )


@mcp.tool()
def rlm_analyze_file(
    file_path: str,
    query: str,
    max_depth: int = 1,
) -> str:
    """
    Analyze a file using RLM recursive processing.

    RLM treats the file content as an external environment variable and uses
    programmatic chunking + recursive sub-LLM calls to process content that
    would exceed normal context windows.

    Best for:
    - Large files (>100K tokens)
    - Information-dense aggregation tasks ("count all X", "find all Y")
    - Codebase analysis requiring examination of most/all content
    - Tasks where every entry matters (not just needle-in-haystack)

    NOT ideal for:
    - Small files where direct LLM processing works
    - Simple retrieval/search tasks (RAG may be better)
    - Tasks requiring holistic understanding without chunking

    Args:
        file_path: Absolute path to the file to analyze
        query: The question or task to perform on the file content
        max_depth: Recursion depth (0=no sub-calls, 1=one level of sub-LLM calls)

    Returns:
        The RLM's response after analyzing the file
    """
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            context = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    char_count = len(context)
    estimated_tokens = char_count // 4  # rough estimate

    rlm = _create_rlm(max_depth=max_depth)

    try:
        result = rlm.completion(prompt=context, root_prompt=query)
        return (
            f"Response:\n{result.response}\n\n"
            f"---\n"
            f"File: {file_path}\n"
            f"Size: {char_count:,} chars (~{estimated_tokens:,} tokens)\n"
            f"Execution time: {result.execution_time:.1f}s"
        )
    except Exception as e:
        return f"Error during RLM processing: {e}"


@mcp.tool()
def rlm_analyze_directory(
    directory_path: str,
    file_pattern: str,
    query: str,
    max_depth: int = 1,
    max_files: int = 100,
) -> str:
    """
    Analyze all matching files in a directory using RLM.

    Combines all matching files into a single context and processes with RLM.
    Useful for codebase-wide analysis, documentation processing, or any task
    requiring examination of multiple files.

    Args:
        directory_path: Path to the directory to analyze
        file_pattern: Glob pattern to match files (e.g., "**/*.py", "*.md", "src/**/*.ts")
        query: The question or task to perform across all matched files
        max_depth: Recursion depth for RLM processing
        max_files: Maximum number of files to include (for safety)

    Returns:
        The RLM's response after analyzing the matched files
    """
    if not os.path.isabs(directory_path):
        directory_path = os.path.abspath(directory_path)

    if not os.path.isdir(directory_path):
        return f"Error: Directory not found: {directory_path}"

    # Find matching files
    pattern = os.path.join(directory_path, file_pattern)
    files = glob.glob(pattern, recursive=True)

    # Filter to actual files (not directories)
    files = [f for f in files if os.path.isfile(f)]

    if not files:
        return f"Error: No files matching pattern '{file_pattern}' found in {directory_path}"

    if len(files) > max_files:
        return (
            f"Error: Found {len(files)} files, exceeds max_files={max_files}. "
            f"Use a more specific pattern or increase max_files."
        )

    # Build combined context
    context_parts = []
    total_chars = 0

    for file_path in sorted(files):
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            rel_path = os.path.relpath(file_path, directory_path)
            context_parts.append(f"=== {rel_path} ===\n{content}")
            total_chars += len(content)
        except Exception as e:
            context_parts.append(f"=== {file_path} ===\n[Error reading file: {e}]")

    combined_context = "\n\n".join(context_parts)
    estimated_tokens = total_chars // 4

    rlm = _create_rlm(max_depth=max_depth)

    try:
        result = rlm.completion(prompt=combined_context, root_prompt=query)
        return (
            f"Response:\n{result.response}\n\n"
            f"---\n"
            f"Directory: {directory_path}\n"
            f"Pattern: {file_pattern}\n"
            f"Files analyzed: {len(files)}\n"
            f"Total size: {total_chars:,} chars (~{estimated_tokens:,} tokens)\n"
            f"Execution time: {result.execution_time:.1f}s"
        )
    except Exception as e:
        return f"Error during RLM processing: {e}"


@mcp.tool()
def rlm_analyze_text(
    text: str,
    query: str,
    max_depth: int = 1,
) -> str:
    """
    Analyze provided text using RLM recursive processing.

    Use this when you already have the text content and want to process it
    with RLM's recursive approach. Note: the text is passed through the MCP
    protocol, so very large texts may be better handled via rlm_analyze_file.

    Args:
        text: The text content to analyze
        query: The question or task to perform on the text
        max_depth: Recursion depth (0=no sub-calls, 1=one level)

    Returns:
        The RLM's response after analyzing the text
    """
    char_count = len(text)
    estimated_tokens = char_count // 4

    rlm = _create_rlm(max_depth=max_depth)

    try:
        result = rlm.completion(prompt=text, root_prompt=query)
        return (
            f"Response:\n{result.response}\n\n"
            f"---\n"
            f"Input size: {char_count:,} chars (~{estimated_tokens:,} tokens)\n"
            f"Execution time: {result.execution_time:.1f}s"
        )
    except Exception as e:
        return f"Error during RLM processing: {e}"


@mcp.tool()
def rlm_info() -> str:
    """
    Get information about the RLM MCP server configuration.

    Returns details about available models, configuration, and usage guidance.
    """
    return f"""RLM MCP Server Configuration
=============================

Default Model: {DEFAULT_MODEL}
Default Sub-Model: {DEFAULT_SUB_MODEL}
API Key Configured: {'Yes' if ANTHROPIC_API_KEY else 'No (set ANTHROPIC_API_KEY)'}

Available Tools:
- rlm_analyze_file: Process a single large file
- rlm_analyze_directory: Process multiple files matching a pattern
- rlm_analyze_text: Process text content directly
- rlm_info: This help message

When to Use RLM:
* Large files (>100K tokens) that exceed context windows
* Information-dense aggregation (count all X, find all pairs Y)
* Codebase-wide analysis requiring most/all code
* Tasks where context rot would hurt quality

When NOT to Use RLM:
* Small files where direct processing works
* Simple needle-in-haystack retrieval
* Tasks requiring fast responses (RLM adds latency)
* Tasks requiring holistic understanding without chunking

Environment Variables:
- RLM_MODEL: Override default model (currently: {DEFAULT_MODEL})
- RLM_SUB_MODEL: Override sub-call model (currently: {DEFAULT_SUB_MODEL})
- ANTHROPIC_API_KEY: Required for Anthropic backend
"""


def main():
    """Entry point for the rlm-mcp command."""
    mcp.run()


if __name__ == "__main__":
    main()
