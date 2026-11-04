"""
Tests for search_api.py, against the real vectors and model.

    pip install pytest
    python -m pytest test_api.py -v -s

Loads the engine once (about a minute), then every test reuses it.
"""

import dataclasses
import threading
import time

import pytest
from fastapi.testclient import TestClient

import search_api
from search_api import Engine, Settings, create_app

IP = {"CF-Connecting-IP": "203.0.113.7"}
TOPIC = "how to start a startup while in college"


@pytest.fixture(scope="session")
def engine():
    eng = Engine(Settings(env="dev"))
    eng.load()
    return eng


def make_client(engine, **overrides):
    s = dataclasses.replace(engine.s, **overrides)
    return TestClient(create_app(s, engine))


@pytest.fixture()
def dev(engine):
    return make_client(engine, env="dev")


@pytest.fixture()
def prod(engine):
    return make_client(engine, env="prod")


def test_health_503_until_loaded():
    c = TestClient(create_app(Settings(env="dev"), Engine(Settings(env="dev"))))
    assert c.get("/health").status_code == 503
    assert c.get("/search", params={"q": "x"}).status_code == 503


def test_health_ok_when_loaded(dev):
    r = dev.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["articles"] == 202450


def test_search_shape_and_order(dev):
    r = dev.get("/search", params={"q": "interpretability of neural networks"})
    assert r.status_code == 200
    body = r.json()
    res = body["results"]
    assert len(res) == 10 and set(res[0]) == {"id", "title", "url", "score"}
    scores = [x["score"] for x in res]
    assert scores == sorted(scores, reverse=True)
    assert "timing_ms" in body and body["timing_ms"]["embed"] > 0
    print("\n  timing:", body["timing_ms"], "| top:", res[0]["title"][:60])


def test_duplicates_combined(engine, dev):
    res = dev.get("/search", params={"q": TOPIC}).json()["results"]
    norm = lambda t: " ".join(t.lower().split())
    # four URL variants of the same essay come back as one result
    assert sum("student's guide to startups" in norm(x["title"]) for x in res) == 1
    # a title may repeat only when the pages really differ (e.g. two different lectures)
    row_of = {int(i): k for k, i in enumerate(engine.ids)}
    for i, a in enumerate(res):
        for b in res[i + 1:]:
            if norm(a["title"]) == norm(b["title"]):
                cos = float(engine.matrix[row_of[a["id"]]] @ engine.matrix[row_of[b["id"]]])
                assert cos <= engine.s.dup_cosine, (a["url"], b["url"], cos)


def test_curius_pages_excluded(dev):
    res = dev.get("/search", params={"q": "curius onboarding welcome to curius"}).json()["results"]
    assert res and all("curius.app" not in x["url"] for x in res)


def test_cache_hit_and_normalization(dev):
    q1 = "  Habits   of HIGHLY productive people "
    first = dev.get("/search", params={"q": q1}).json()
    again = dev.get("/search", params={"q": "habits of highly productive people"}).json()
    assert first["cached"] is False and again["cached"] is True
    assert first["results"] == again["results"] and first["query"] == "habits of highly productive people"


def test_long_query_truncated(dev):
    r = dev.get("/search", params={"q": "startups " * 300})
    assert r.status_code == 200 and len(r.json()["query"]) <= 500


def test_empty_and_missing_query(dev):
    assert dev.get("/search", params={"q": "    "}).status_code == 400
    assert dev.get("/search").status_code == 422


def test_prod_requires_cf_header_and_hides_docs(prod):
    assert prod.get("/search", params={"q": "writing"}).status_code == 400
    r = prod.get("/search", params={"q": "writing advice"}, headers=IP)
    assert r.status_code == 200 and "timing_ms" not in r.json()
    assert prod.get("/docs").status_code == 404


def test_dev_falls_back_to_connection_ip(dev):
    assert dev.get("/search", params={"q": "writing advice"}).status_code == 200


def test_rate_limit_per_cf_ip(engine):
    c = make_client(engine, env="prod", rate_per_min=3)
    codes = [c.get("/search", params={"q": f"rate test {i}"}, headers=IP).status_code for i in range(4)]
    assert codes == [200, 200, 200, 429]
    r = c.get("/search", params={"q": "rate test 9"}, headers=IP)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    other = c.get("/search", params={"q": "rate test 1"}, headers={"CF-Connecting-IP": "198.51.100.9"})
    assert other.status_code == 200  # a different visitor is unaffected


def test_daily_limit(engine):
    c = make_client(engine, env="prod", rate_per_min=100, rate_per_day=2)
    codes = [c.get("/search", params={"q": f"daily {i}"}, headers=IP).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_503_when_all_model_copies_busy(engine):
    c = make_client(engine, env="prod")
    held = [engine.acquire() for _ in range(engine.s.max_concurrent)]
    assert all(m is not None for m in held)
    try:
        r = c.get("/search", params={"q": "busy server test"}, headers=IP)
        assert r.status_code == 503 and r.headers["Retry-After"] == "2"
        # a cached query is still answered while every embedder is busy
        engine.release(held.pop())
        assert c.get("/search", params={"q": "busy server test"}, headers=IP).status_code == 200
        engine.release(held.pop())
        held = [engine.acquire(), engine.acquire()]
        assert c.get("/search", params={"q": "busy server test"}, headers=IP).json()["cached"] is True
    finally:
        for m in held:
            engine.release(m)


def test_at_most_two_embeddings_at_once(engine):
    c = make_client(engine, env="prod", rate_per_min=100)
    peak, running, lock, codes = [0], [0], threading.Lock(), []
    original = Engine.embed

    def spy(self, text, model=None):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        try:
            return original(self, text, model=model)
        finally:
            with lock:
                running[0] -= 1

    Engine.embed = spy
    try:
        def go(i):
            codes.append(c.get("/search", params={"q": f"parallel query {i}"}, headers=IP).status_code)
        threads = [threading.Thread(target=go, args=(i,)) for i in range(5)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    finally:
        Engine.embed = original
    assert set(codes) <= {200, 503} and codes.count(200) >= 2
    assert peak[0] <= 2
    print(f"\n  5 simultaneous requests -> {codes.count(200)} served, {codes.count(503)} told to retry; peak concurrent embeddings {peak[0]}")


def test_cors_prod(prod):
    ok = prod.get("/health", headers={"Origin": "https://vishal.fast"})
    assert ok.headers.get("access-control-allow-origin") == "https://vishal.fast"
    assert "Retry-After" in ok.headers.get("access-control-expose-headers", "")  # lets the page read it
    for bad in ("http://localhost:3000", "https://evil.example"):
        r = prod.get("/health", headers={"Origin": bad})
        assert "access-control-allow-origin" not in r.headers


def test_cors_dev(dev):
    ok = dev.get("/health", headers={"Origin": "http://localhost:3000"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    r = dev.get("/health", headers={"Origin": "https://vishal.fast"})
    assert "access-control-allow-origin" not in r.headers


def test_queue_wait_serves_request_when_a_copy_frees_up(engine):
    c = make_client(engine, env="prod", queue_wait_s=3.0)
    held = [engine.acquire() for _ in range(engine.s.max_concurrent)]
    assert all(m is not None for m in held)
    timer = threading.Timer(0.5, lambda: [engine.release(m) for m in held])
    timer.start()
    t0 = time.time()
    r = c.get("/search", params={"q": "queue wait test one"}, headers=IP)
    waited = time.time() - t0
    timer.join()
    assert r.status_code == 200 and waited >= 0.5


def test_queue_wait_gives_up_with_503(engine):
    c = make_client(engine, env="prod", queue_wait_s=0.3)
    held = [engine.acquire() for _ in range(engine.s.max_concurrent)]
    try:
        t0 = time.time()
        r = c.get("/search", params={"q": "queue wait test two"}, headers=IP)
        assert r.status_code == 503 and r.headers["Retry-After"] == "2"
        assert 0.25 <= time.time() - t0 < 2.0
    finally:
        for m in held:
            engine.release(m)
