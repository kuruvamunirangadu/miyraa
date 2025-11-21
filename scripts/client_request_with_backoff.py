"""Client helper for POST requests with exponential backoff and timeout.

Usage:
  python scripts/client_request_with_backoff.py --url http://localhost:8000/predict --text "hello"

This implements jittered exponential backoff and a per-request timeout.
"""
import argparse
import time
import random
import requests


def post_with_backoff(url, json_payload, timeout=5, max_retries=5, base_delay=0.5, backoff_factor=2.0):
    attempt = 0
    while True:
        try:
            resp = requests.post(url, json=json_payload, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                raise
            # Exponential backoff with jitter
            delay = base_delay * (backoff_factor ** (attempt - 1))
            jitter = random.uniform(0, delay * 0.1)
            sleep_time = delay + jitter
            print(f"Request failed (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--max-retries", type=int, default=5)
    args = p.parse_args()

    payload = {"text": args.text}
    r = post_with_backoff(args.url, payload, timeout=args.timeout, max_retries=args.max_retries)
    print(f"Status: {r.status_code}")
    print(r.text)
