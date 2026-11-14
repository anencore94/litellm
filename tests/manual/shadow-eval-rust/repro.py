import asyncio
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
from threading import Thread
from types import SimpleNamespace

import httpx
import litellm
import psycopg
import uvicorn
from psycopg import sql
from litellm.proxy import proxy_server
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.integrations.shadow_eval_logger import ShadowEvalLogger

BASE = Path(__file__).resolve().parent
BIN = Path(os.environ["SHADOW_TEST_GATEWAY"]).resolve()
PG = os.environ["SHADOW_TEST_PG"]
calls, visits, funnel = [], [], []


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        calls.append((self.path, data["model"]))
        if self.path == "/v1/messages":
            response = dict(id="msg_local", type="message", role="assistant", model=data["model"],
                            content=[dict(type="text", text="Primary answer")], stop_reason="end_turn",
                            stop_sequence=None, usage=dict(input_tokens=12, output_tokens=4))
        else:
            assert self.path == "/v1/chat/completions", self.path
            content = ('{"preference":"tie","confidence":1.0}'
                       if data["model"] == "gpt-4o" else "Shadow answer")
            response = dict(id="chatcmpl-local", object="chat.completion", created=1, model=data["model"],
                            choices=[dict(index=0, message=dict(role="assistant", content=content),
                                          finish_reason="stop")],
                            usage=dict(prompt_tokens=12, completion_tokens=4, total_tokens=16))
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class JobTable:
    async def find_many(self, *, where):
        print("JOB READ", where, flush=True)
        with psycopg.connect(PG, row_factory=psycopg.rows.dict_row) as conn:
            rows = conn.execute('SELECT * FROM "LiteLLM_ShadowEvalJob" WHERE ends_at > now()').fetchall()
            return [SimpleNamespace(**r) for r in rows]


class AttemptTable:
    async def group_by(self, **kwargs):
        with psycopg.connect(PG) as conn:
            rows = conn.execute(
                'SELECT job_id, count(*), sum(judge_cost), sum(shadow_cost), sum(shadow_classifier_cost) '
                'FROM "LiteLLM_ShadowEvalAttempt" GROUP BY job_id'
            ).fetchall()
        return [dict(job_id=j, _count={"_all": n}, _sum=dict(judge_cost=a, shadow_cost=b,
                     shadow_classifier_cost=c)) for j, n, a, b, c in rows]

    async def create(self, *, data):
        with psycopg.connect(PG) as conn:
            query = sql.SQL('INSERT INTO "LiteLLM_ShadowEvalAttempt" ({}) VALUES ({})').format(
                sql.SQL(", ").join(map(sql.Identifier, data)), sql.SQL(", ").join(sql.Placeholder() for _ in data))
            conn.execute(query, tuple(data.values()))


async def identity():
    return UserAPIKeyAuth(api_key="local-hashed-key", team_id="local-team", user_id="local-user")


async def traced_app(scope, receive, send):
    if scope["type"] == "http" and scope["path"] == "/v1/messages":
        headers = dict(scope["headers"])
        visits.append(headers.get(b"x-litellm-rust-peer", b"").decode())
    await proxy_server.app(scope, receive, send)


def attempts():
    with psycopg.connect(PG, row_factory=psycopg.rows.dict_row) as conn:
        return conn.execute('SELECT * FROM "LiteLLM_ShadowEvalAttempt" ORDER BY row_id').fetchall()


def model_params(model, url):
    return dict(model=model, api_base=url, api_key="local-fixture")


async def main(stub_port):
    root = f"http://127.0.0.1:{stub_port}"
    config_path = BASE / "config.yaml"
    config_path.write_text(json.dumps(dict(
        model_list=[
            dict(model_name="primary", litellm_params=model_params("anthropic/claude-sonnet-4-6", root)),
            dict(model_name="shadow", litellm_params=model_params("openai/gpt-4o-mini", root + "/v1")),
            dict(model_name="judge", litellm_params=model_params("openai/gpt-4o", root + "/v1")),
        ],
        router_settings=dict(num_retries=0), general_settings=dict(master_key="local-parity-master"),
    )))
    with psycopg.connect(PG) as conn:
        conn.execute('CREATE TABLE "LiteLLM_ShadowEvalJob" (id text, target_type text, target_id text, '
                     'router_name text, judge_model text, shadow_percentage float8, max_turns int, ends_at timestamptz)')
        conn.execute('INSERT INTO "LiteLLM_ShadowEvalJob" VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
                     ("job-local", "team", "local-team", "shadow", "judge", 100.0, 8,
                      datetime.now(timezone.utc) + timedelta(hours=1)))
        conn.execute('CREATE TABLE "LiteLLM_ShadowEvalAttempt" (row_id bigserial, job_id text, request_id text, '
                     'router_name text, outcome text, tier text, real_model text, shadow_model text, confidence float8, '
                     'judge_cost float8, shadow_cost float8, shadow_classifier_cost float8, real_cost float8, '
                     'real_classifier_cost float8, real_cache_hit bool, error text)')
    await proxy_server.initialize(config=str(config_path))
    prisma = SimpleNamespace(db=SimpleNamespace(litellm_shadowevaljob=JobTable(),
                                                litellm_shadowevalattempt=AttemptTable()))
    class ObservedLogger(ShadowEvalLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time, **rest):
            payload = kwargs.get("standard_logging_object") or {}
            meta = payload.get("metadata") or {}
            from litellm.integrations import shadow_eval_logger as m
            raw = m.get_litellm_metadata_from_kwargs(dict(kwargs)) or {}
            print("GATES", payload.get("id"), m.should_redact_message_logging(dict(kwargs)), raw.get(m.INTERNAL_CALL_ORIGIN_METADATA_KEY), flush=True)
            print("HOOK", payload.get("call_type"), type(response_obj), {k:v for k,v in meta.items() if k in ("user_api_key_hash", "user_api_key_team_id", "user_api_key_user_id")}, flush=True)
            return await super().async_log_success_event(kwargs, response_obj, start_time, end_time, **rest)

    logger = ObservedLogger(prisma_provider=lambda: prisma,
                              funnel_recorder=lambda job, stage: funnel.append((job, stage)))
    litellm.logging_callback_manager.add_litellm_callback(logger)
    litellm.logging_callback_manager.add_litellm_async_success_callback(logger)
    proxy_server.app.dependency_overrides[user_api_key_auth] = identity

    py_port, rust_port, native_port = port(), port(), port()
    python = uvicorn.Server(uvicorn.Config(traced_app, host="127.0.0.1", port=py_port, lifespan="off"))
    task = asyncio.create_task(python.serve())
    while not python.started:
        await asyncio.sleep(.05)
    base_env = dict(os.environ)
    forward = dict(LITELLM_FORWARD_UPSTREAM=f"http://127.0.0.1:{py_port}",
                   LITELLM_FORWARD_HOST="127.0.0.1", LITELLM_FORWARD_PORT=str(rust_port),
                   LITELLM_FORWARD_UPSTREAM_KEEPALIVE="5", LITELLM_FORWARD_SHUTDOWN_DRAIN="2")
    native = dict(HOST="127.0.0.1", PORT=str(native_port), LITELLM_CONFIG=str(config_path))
    procs = []
    body = dict(model="primary", max_tokens=32, stream=False,
                messages=[dict(role="user", content="Say hello.")])
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            for name, listen, extra in (("python", py_port, None),
                                        ("rust-forward", rust_port, forward), ("rust-native", native_port, native)):
                url = f"http://127.0.0.1:{listen}"
                if extra:
                    with (BASE / f"{name}.log").open("w") as log:
                        proc = subprocess.Popen([str(BIN)], stdin=subprocess.PIPE, stdout=log,
                                                stderr=log, env={**base_env, **extra})
                    procs.append(proc)
                    for _ in range(100):
                        try:
                            await client.get(url + "/health/liveliness")
                            break
                        except httpx.ConnectError:
                            if proc.poll() is not None:
                                raise RuntimeError(f"{name} exited: {proc.returncode}")
                            await asyncio.sleep(.05)
                    else:
                        raise RuntimeError(f"{name} did not bind")
                old_rows, old_visits, old_calls, old_funnel = len(attempts()), len(visits), len(calls), len(funnel)
                response = await client.post(url + "/v1/messages", json=body, headers={"Authorization":"Bearer local-parity-master"})
                for _ in range(40):
                    await asyncio.sleep(.1)
                    if len(attempts()) > old_rows and logger._inflight_shadow_tasks == 0:
                        break
                await asyncio.sleep(.3)
                print("RESULT", json.dumps(dict(lane=name, status=response.status_code, reply=response.text,
                      python_visits=visits[old_visits:], provider_calls=calls[old_calls:],
                      attempts=attempts()[old_rows:], funnel=funnel[old_funnel:]), default=str), flush=True)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(4)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        python.should_exit = True
        await task
        await litellm.close_litellm_async_clients()


with ThreadingHTTPServer(("127.0.0.1", 0), Stub) as stub:
    worker = Thread(target=stub.serve_forever, daemon=True)
    worker.start()
    try:
        asyncio.run(main(stub.server_port))
    finally:
        stub.shutdown()
        worker.join()
