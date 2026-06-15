import os
import time

from flask import Flask, jsonify
import redis
import requests


app = Flask(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def burn_cpu_if_enabled():
    if os.getenv("ENABLE_CPU_BURN", "false").lower() != "true":
        return

    burn_ms = int(os.getenv("CPU_BURN_MS", "30"))
    deadline = time.perf_counter() + burn_ms / 1000
    value = 0
    while time.perf_counter() < deadline:
        value += 1
    return value


@app.get("/api/ping")
def ping():
    burn_cpu_if_enabled()
    print("received /api/ping request", flush=True)
    return jsonify(status="ok")


@app.get("/api/redis")
def redis_check():
    client = get_redis_client()
    client.incr("visit_count")
    return jsonify(status="ok", visit_count=int(client.get("visit_count")))


@app.get("/api/extra-package")
def extra_package_check():
    return jsonify(status="ok", requests_version=requests.__version__)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
