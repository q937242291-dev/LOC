"""Provider sampling and request seeds, using in-memory endpoint responses."""
from dataclasses import replace
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error

from locus.config import ModelSettings, Settings, load_config
from locus.ledger import Ledger
from locus.models import HTTPModel, ModelError


ROOT = Path(__file__).resolve().parents[1]


class SamplingTests(unittest.TestCase):
    def client(self, *, seed=17, mode="request", extra=None):
        settings = Settings(seed=seed)
        config = ModelSettings(base_url="http://localhost:1234/v1", seed_mode=mode,
                               extra_body=extra or {})
        return HTTPModel(config, settings, Ledger(settings), len)

    def response(self, request, **kwargs):
        self.bodies.append(json.loads(request.data))
        return io.BytesIO(json.dumps(dict(model="served-snapshot", system_fingerprint="fp-test",
            choices=[dict(message=dict(content='{"ok":true}'), finish_reason="stop")],
            usage=dict(prompt_tokens=1, completion_tokens=1))).encode())

    def test_role_local_seed_sequence_pairs_common_roles(self):
        self.bodies = []
        with patch("locus.models.urllib.request.urlopen", side_effect=self.response):
            left, right = self.client(), self.client()
            left.ask("ground", "JSON", "one")
            left.ask("extra", "JSON", "one")
            left.ask("ground", "JSON", "two")
            right.ask("ground", "JSON", "one")
            right.ask("ground", "JSON", "two")
        seeds = [b["seed"] for b in self.bodies]
        self.assertEqual(seeds[0], seeds[3])
        self.assertEqual(seeds[2], seeds[4])
        self.assertNotEqual(seeds[0], seeds[2])
        self.assertTrue(all(type(value) is int and 0 <= value < 2**31 for value in seeds))
        metadata = left.request_metadata[0]
        self.assertEqual(metadata["served_model"], "served-snapshot")
        self.assertEqual(metadata["system_fingerprint"], "fp-test")
        self.assertEqual(metadata["requested_seed"], seeds[0])

    def test_off_mode_does_not_send_a_seed(self):
        self.bodies = []
        model = self.client(mode="off")
        with patch("locus.models.urllib.request.urlopen", side_effect=self.response):
            model.ask("patch", "JSON", "one")
        self.assertNotIn("seed", self.bodies[0])
        self.assertIsNone(model.request_metadata[0]["requested_seed"])

    def test_seed_override_cannot_bypass_explicit_policy(self):
        model = self.client(extra={"seed": 99})
        with patch("locus.models.urllib.request.urlopen") as endpoint:
            with self.assertRaises(ValueError):
                model.ask("patch", "JSON", "one")
        endpoint.assert_not_called()

    def test_unsupported_seed_is_not_retried_without_seed(self):
        model = self.client()
        failure = urllib.error.HTTPError("http://localhost", 400, "bad parameter", {}, io.BytesIO(b"{}"))
        with patch("locus.models.urllib.request.urlopen", side_effect=failure) as endpoint:
            with self.assertRaises(ModelError):
                model.ask("patch", "JSON", "one")
        self.assertEqual(endpoint.call_count, 1)
        self.assertEqual(len(model.request_metadata), 1)
        self.assertFalse(model.request_metadata[0]["response_received"])

    def test_native_anthropic_does_not_claim_seed_support(self):
        with self.assertRaises(ValueError):
            ModelSettings(provider="anthropic", seed_mode="request").validate()
        with self.assertRaises(ValueError):
            ModelSettings(seed_mode="auto").validate()

    def test_deepseek_paper_config_keeps_temperature_active(self):
        settings, config = load_config(ROOT / "configs/deepseek.toml")
        config = replace(config, base_url="http://localhost:1234/v1")
        self.bodies = []
        with patch("locus.models.urllib.request.urlopen", side_effect=self.response):
            HTTPModel(config, settings, Ledger(settings), len).ask("patch", "JSON", "one")
        self.assertEqual(self.bodies[0]["thinking"], {"type": "disabled"})
        self.assertEqual(self.bodies[0]["temperature"], .7)
        self.assertEqual(self.bodies[0]["top_p"], 1.)


if __name__ == "__main__":
    unittest.main()
