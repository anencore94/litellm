# Shadow Eval through the Rust gateway

See [the tracking issue](https://github.com/anencore94/litellm/issues/10) for the observed behavior and proposed admission test. This harness exercises the Python proxy, the Rust forwarding listener, and the standalone Rust listener with the same non-streaming Messages request. It uses local fixtures for authentication and providers, and a small PostgreSQL adapter for the evaluation tables.

Use Python 3.12 with `litellm[proxy]==1.111.0.dev1` and `psycopg[binary]`. Set `SHADOW_TEST_GATEWAY` to the matching `litellm-gateway` executable. Set `SHADOW_TEST_PG` to the connection string for a fresh, disposable local UTF-8 PostgreSQL database. The script creates fixture tables in that database and writes a synthetic model config next to itself.

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python repro.py
```

The recorded run returned HTTP 200 in all three cases and wrote 1, 1, and 0 attempt rows respectively. The forwarder includes the Rust peer header on the Python hop. The native run has no Python hop or connection to the Python evaluation fixture. The exact source commit was inspected separately. Its runtime needs a matching native extension.

Once the gateway supports per-request native admission, add an integration case that asserts a matching one-arm job triggers one primary, one shadow, and one judge call, with exactly one persisted attempt. Keep the forwarding decision before the first provider call.
