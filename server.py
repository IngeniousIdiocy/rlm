#!/usr/bin/env python3
"""OpenAI API-compatible HTTP server for RLM completions."""

import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from rlm import RLM

MODEL = os.environ.get("RLM_MODEL", "claude-opus-4-5-20251101")

rlm = RLM(
    backend="anthropic",
    backend_kwargs={
        "model_name": MODEL,
        "api_key": os.environ["ANTHROPIC_API_KEY"],
    },
    environment="local",
    max_depth=1,
    verbose=True,
)


def truncate(s, n=50):
    s = s.replace("\n", "\\n")
    return s[:n] + "..." if len(s) > n else s


def messages_to_prompt(messages):
    """Convert OpenAI messages array to a single prompt string."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    return "\n\n".join(parts)


class RLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)

        # Extract messages and convert to prompt
        messages = data.get("messages", [])
        if not messages:
            # Fallback for simple prompt field
            prompt = data.get("prompt", "")
        else:
            prompt = messages_to_prompt(messages)

        print(f"[{self.log_date_time_string()}] >>> {truncate(prompt)}")

        result = rlm.completion(prompt)
        response_text = result.response
        print(f"[{self.log_date_time_string()}] <<< {truncate(response_text)}")

        # Build OpenAI-compatible response
        usage = result.usage_summary.to_dict() if result.usage_summary else {}
        total_input = sum(m.get("total_input_tokens", 0) for m in usage.get("model_usage_summaries", {}).values())
        total_output = sum(m.get("total_output_tokens", 0) for m in usage.get("model_usage_summaries", {}).values())

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": total_input,
                "completion_tokens": total_output,
                "total_tokens": total_input + total_output,
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def do_GET(self):
        # Minimal /v1/models endpoint for compatibility
        if self.path in ("/v1/models", "/models"):
            response = {
                "object": "list",
                "data": [
                    {
                        "id": MODEL,
                        "object": "model",
                        "owned_by": "rlm",
                    }
                ],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress default HTTP logging


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8787))
    server = HTTPServer(("0.0.0.0", port), RLMHandler)
    print(f"RLM server running on http://localhost:{port}")
    print(f"Model: {MODEL}")
    print(f"OpenAI-compatible endpoint: POST /v1/chat/completions")
    server.serve_forever()
