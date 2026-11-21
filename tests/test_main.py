"""Regression tests for the proxy, run with `python3 -m unittest discover -s tests`.

The point of these is the security property this project exists for: a client key
must not be able to mint more client keys. Everything else here is a behaviour
that used to be quietly wrong (token usage always 0, max_tokens mapped to the
context window instead of the output limit, keys lost on restart).
"""

import importlib
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ADMIN = "admin-secret-for-tests"


def fresh_app(keys_file, admin=ADMIN):
    """Reload the module so module-level env reads re-run."""
    os.environ["KEYS_FILE"] = keys_file
    if admin is None:
        os.environ.pop("ADMIN_API_KEY", None)
    else:
        os.environ["ADMIN_API_KEY"] = admin
    sys.modules.pop("main", None)
    return importlib.import_module("main")


class KeyManagementAuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.keys_file = os.path.join(self.tmp.name, "api_keys.json")
        self.main = fresh_app(self.keys_file)
        self.client = self.main.app

    def tearDown(self):
        self.tmp.cleanup()
        sys.modules.pop("main", None)

    def _call(self, method, path, **kw):
        from fastapi.testclient import TestClient

        with TestClient(self.client) as c:
            return getattr(c, method)(path, **kw)

    def test_key_management_is_closed_without_a_credential(self):
        """The bug this file exists for: /api/keys used to be wide open."""
        self.assertEqual(self._call("post", "/api/keys").status_code, 401)
        self.assertEqual(self._call("get", "/api/keys").status_code, 401)
        self.assertEqual(self._call("delete", "/api/keys/whatever").status_code, 401)

    def test_a_client_key_cannot_manage_keys(self):
        created = self._call(
            "post", "/api/keys", headers={"Authorization": f"Bearer {ADMIN}"}
        )
        self.assertEqual(created.status_code, 201)
        client_key = created.json()["api_key"]

        denied = self._call(
            "post", "/api/keys", headers={"Authorization": f"Bearer {client_key}"}
        )
        self.assertEqual(denied.status_code, 401)

    def test_admin_can_create_and_delete(self):
        h = {"Authorization": f"Bearer {ADMIN}"}
        created = self._call("post", "/api/keys", headers=h).json()["api_key"]
        listed = self._call("get", "/api/keys", headers=h).json()["keys"]
        self.assertEqual(len(listed), 1)
        # the list must not hand the secret back out
        self.assertNotIn(created, json.dumps(listed))

        self.assertEqual(self._call("delete", f"/api/keys/{created}", headers=h).status_code, 200)
        self.assertEqual(self._call("get", "/api/keys", headers=h).json()["keys"], [])

    def test_created_key_authorises_the_proxy(self):
        client_key = self._call(
            "post", "/api/keys", headers={"Authorization": f"Bearer {ADMIN}"}
        ).json()["api_key"]

        tags = {"models": [{"name": "llama3:latest"}]}
        with mock.patch.object(self.main.requests, "get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = tags
            get.return_value.raise_for_status.return_value = None

            ok = self._call(
                "get", "/v1/models", headers={"Authorization": f"Bearer {client_key}"}
            )
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(ok.json()["data"][0]["id"], "llama3:latest")

            anon = self._call("get", "/v1/models")
            self.assertEqual(anon.status_code, 401)


class PersistenceTests(unittest.TestCase):
    def test_keys_and_admin_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            keys_file = os.path.join(tmp, "api_keys.json")
            first = fresh_app(keys_file)
            from fastapi.testclient import TestClient

            with TestClient(first.app) as c:
                client_key = c.post(
                    "/api/keys", headers={"Authorization": f"Bearer {ADMIN}"}
                ).json()["api_key"]

            # Restart with no ADMIN_API_KEY in the environment: the persisted
            # admin key has to come back, or the operator is locked out.
            second = fresh_app(keys_file, admin=None)
            self.assertEqual(second.ADMIN_API_KEY, ADMIN)
            self.assertIn(client_key, second.VALID_API_KEYS)

            sys.modules.pop("main", None)


class PayloadMappingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.main = fresh_app(os.path.join(self.tmp.name, "api_keys.json"))

    def tearDown(self):
        self.tmp.cleanup()
        sys.modules.pop("main", None)

    def test_max_tokens_maps_to_num_predict_not_num_ctx(self):
        req = self.main.ChatCompletionRequest(
            model="llama3", messages=[self.main.ChatMessage(role="user", content="hi")],
            max_tokens=50,
        )
        payload = self.main._ollama_payload(req)
        self.assertEqual(payload["options"]["num_predict"], 50)
        self.assertNotIn("num_ctx", payload["options"])

    def test_usage_reports_the_counts_ollama_actually_returns(self):
        from fastapi.testclient import TestClient

        with TestClient(self.main.app) as c:
            client_key = c.post(
                "/api/keys", headers={"Authorization": f"Bearer {ADMIN}"}
            ).json()["api_key"]

            with mock.patch.object(self.main.requests, "post") as post:
                post.return_value.status_code = 200
                post.return_value.raise_for_status.return_value = None
                post.return_value.json.return_value = {
                    "message": {"role": "assistant", "content": "hello"},
                    "done": True,
                    "prompt_eval_count": 12,
                    "eval_count": 7,
                }
                r = c.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {client_key}"},
                    json={
                        "model": "llama3",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.json()["usage"],
            {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        )

    def test_the_admin_key_is_not_a_client_key(self):
        """Two credentials, two jobs: the admin key must not reach Ollama."""
        from fastapi.testclient import TestClient

        with TestClient(self.main.app) as c:
            r = c.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {ADMIN}"},
                json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
            )
        self.assertEqual(r.status_code, 401)


class ParameterMappingTests(unittest.TestCase):
    """Ollama only reads sampling parameters from `options`, never top-level."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.main = fresh_app(os.path.join(self.tmp.name, "api_keys.json"))

    def tearDown(self):
        self.tmp.cleanup()
        sys.modules.pop("main", None)

    def _payload(self, **kwargs):
        req = self.main.ChatCompletionRequest(
            model="llama3",
            messages=[self.main.ChatMessage(role="user", content="hi")],
            **kwargs,
        )
        return self.main._ollama_payload(req)

    def test_temperature_lands_in_options_not_top_level(self):
        payload = self._payload(temperature=0.2)
        self.assertEqual(payload["options"]["temperature"], 0.2)
        self.assertNotIn("temperature", payload)

    def test_stop_accepts_a_string_and_a_list(self):
        self.assertEqual(self._payload(stop="END")["options"]["stop"], ["END"])
        self.assertEqual(self._payload(stop=["A", "B"])["options"]["stop"], ["A", "B"])

    def test_seed_and_max_completion_tokens_map_to_options(self):
        payload = self._payload(seed=7, max_completion_tokens=40)
        self.assertEqual(payload["options"]["seed"], 7)
        self.assertEqual(payload["options"]["num_predict"], 40)

    def test_max_tokens_wins_over_max_completion_tokens(self):
        payload = self._payload(max_tokens=50, max_completion_tokens=40)
        self.assertEqual(payload["options"]["num_predict"], 50)


class ErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.main = fresh_app(os.path.join(self.tmp.name, "api_keys.json"))

    def tearDown(self):
        self.tmp.cleanup()
        sys.modules.pop("main", None)

    def test_non_ascii_bearer_header_is_401_not_500(self):
        """compare_digest() raises TypeError on non-ASCII str.

        httpx refuses to build such a request client-side, so drive the auth
        dependency directly — that is the code that must not blow up.
        """
        import asyncio

        self.main.VALID_API_KEYS.add("sk-real-key")
        with self.assertRaises(self.main.HTTPException) as ctx:
            asyncio.run(self.main.get_api_key(auth_header="Bearer pässwörd"))
        self.assertEqual(ctx.exception.status_code, 401)
        with self.assertRaises(self.main.HTTPException) as ctx:
            asyncio.run(self.main.require_admin(auth_header="Bearer pässwörd"))
        self.assertEqual(ctx.exception.status_code, 401)

    def test_ollama_4xx_surfaces_its_message_and_status(self):
        from fastapi.testclient import TestClient

        resp = mock.Mock(status_code=404)
        resp.json.return_value = {"error": "model 'nope' not found"}
        error = self.main.requests.exceptions.HTTPError(response=resp)

        with TestClient(self.main.app) as c:
            client_key = c.post(
                "/api/keys", headers={"Authorization": f"Bearer {ADMIN}"}
            ).json()["api_key"]
            with mock.patch.object(self.main.requests, "post", side_effect=error):
                r = c.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {client_key}"},
                    json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
                )
        self.assertEqual(r.status_code, 404)
        self.assertIn("model 'nope' not found", r.json()["detail"])

    def test_connection_failure_still_reports_502(self):
        from fastapi.testclient import TestClient

        error = self.main.requests.exceptions.ConnectionError("connection refused")
        with TestClient(self.main.app) as c:
            client_key = c.post(
                "/api/keys", headers={"Authorization": f"Bearer {ADMIN}"}
            ).json()["api_key"]
            with mock.patch.object(self.main.requests, "post", side_effect=error):
                r = c.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {client_key}"},
                    json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
                )
        self.assertEqual(r.status_code, 502)
        self.assertIn("unreachable", r.json()["detail"])

    def test_done_false_returns_200_with_null_finish_reason(self):
        from fastapi.testclient import TestClient

        with TestClient(self.main.app) as c:
            client_key = c.post(
                "/api/keys", headers={"Authorization": f"Bearer {ADMIN}"}
            ).json()["api_key"]
            with mock.patch.object(self.main.requests, "post") as post:
                post.return_value.raise_for_status.return_value = None
                post.return_value.json.return_value = {
                    "message": {"role": "assistant", "content": "hi"},
                    "done": False,
                }
                r = c.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {client_key}"},
                    json={"model": "llama3", "messages": [{"role": "user", "content": "hi"}]},
                )
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["choices"][0]["finish_reason"])


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.main = fresh_app(os.path.join(self.tmp.name, "api_keys.json"))

    def tearDown(self):
        self.tmp.cleanup()
        sys.modules.pop("main", None)

    def _stream(self, lines):
        from fastapi.testclient import TestClient

        with TestClient(self.main.app) as c:
            client_key = c.post(
                "/api/keys", headers={"Authorization": f"Bearer {ADMIN}"}
            ).json()["api_key"]
            with mock.patch.object(self.main.requests, "post") as post:
                post.return_value.raise_for_status.return_value = None
                post.return_value.iter_lines.return_value = iter(lines)
                return c.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {client_key}"},
                    json={
                        "model": "llama3",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )

    def test_stream_ends_with_exactly_one_done_marker(self):
        r = self._stream(
            [
                b'{"message": {"role": "assistant", "content": "he"}, "done": false}',
                b'{"message": {"role": "assistant", "content": "llo"}, "done": true}',
            ]
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.text.count("data: [DONE]"), 1)
        self.assertTrue(r.text.rstrip().endswith("data: [DONE]"))
        self.assertIn('"content": "he"', r.text)

    def test_stream_forwards_an_ollama_error_object(self):
        r = self._stream([b'{"error": "model not found"}'])
        self.assertEqual(r.status_code, 200)
        self.assertIn('"error"', r.text)
        self.assertIn("model not found", r.text)
        self.assertEqual(r.text.count("data: [DONE]"), 1)


class ConfigTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("main", None)

    def test_bad_numeric_env_var_names_the_variable(self):
        saved = os.environ.get("LISTEN_PORT")
        os.environ["LISTEN_PORT"] = "not-a-port"
        try:
            sys.modules.pop("main", None)
            with self.assertRaises(SystemExit) as ctx:
                importlib.import_module("main")
            self.assertIn("LISTEN_PORT", str(ctx.exception))
        finally:
            if saved is None:
                os.environ.pop("LISTEN_PORT", None)
            else:
                os.environ["LISTEN_PORT"] = saved

    def test_relative_keys_file_is_anchored_to_the_project(self):
        saved = {name: os.environ.get(name) for name in ("KEYS_FILE", "ADMIN_API_KEY")}
        # Pin the admin key so the import below does not generate one and
        # write api_keys.json into the project directory.
        os.environ["KEYS_FILE"] = "api_keys.json"
        os.environ["ADMIN_API_KEY"] = ADMIN
        try:
            sys.modules.pop("main", None)
            main = importlib.import_module("main")
            self.assertTrue(main.KEYS_FILE.is_absolute())
            self.assertEqual(
                main.KEYS_FILE.parent, pathlib.Path(main.__file__).resolve().parent
            )
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
