"""Regression tests for the proxy, run with `python3 -m unittest discover -s tests`.

The point of these is the security property this project exists for: a client key
must not be able to mint more client keys. Everything else here is a behaviour
that used to be quietly wrong (token usage always 0, max_tokens mapped to the
context window instead of the output limit, keys lost on restart).
"""

import importlib
import json
import os
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


if __name__ == "__main__":
    unittest.main()
