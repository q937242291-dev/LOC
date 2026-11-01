"""Exercise launch templates with the documented context, without upstream execution."""

import subprocess
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from locus.baselines import run_external


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "baselines.toml"


class UpstreamSamplingTests(unittest.TestCase):
    def launch(self, system, context):
        with patch("locus.baselines.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0)) as execute:
            run_external(CONFIG, system, context, cwd=Path("upstream checkout"))
        execute.assert_called_once()
        self.assertEqual(execute.call_args.kwargs["cwd"], Path("upstream checkout"))
        self.assertNotIn("shell", execute.call_args.kwargs)
        return execute.call_args.args[0]

    def test_swe_batch_sends_paper_sampling_with_documented_context(self):
        argv = self.launch("swe-agent", {"model": "openai/gpt-5.2", "workers": 1,
                                         "benchmark": "lite", "output": "outputs/repeat 0"})
        self.assertEqual(argv[:2], ["sweagent", "run-batch"])
        self.assertEqual(float(argv[argv.index("--agent.model.temperature") + 1]), 0.7)
        self.assertEqual(float(argv[argv.index("--agent.model.top_p") + 1]), 1.0)
        self.assertEqual(argv[argv.index("--output_dir") + 1], "outputs/repeat 0")

    def test_mini_retains_base_config_before_recursive_sampling_overrides(self):
        argv = self.launch("mini-swe-agent", {"model": "openai/gpt-5.2", "workers": 1,
                                              "output": "outputs/repeat 0"})
        self.assertEqual(argv[:2], ["mini-extra", "swebench"])
        specs = [argv[index + 1] for index, item in enumerate(argv) if item == "-c"]
        self.assertEqual(specs[0], "swebench.yaml")
        overrides = dict(spec.split("=", 1) for spec in specs[1:])
        self.assertEqual(float(overrides["model.model_kwargs.temperature"]), 0.7)
        self.assertEqual(float(overrides["model.model_kwargs.top_p"]), 1.0)
        self.assertEqual(argv[argv.index("--output") + 1], "outputs/repeat 0")

    def test_openhands_keeps_llm_json_as_positional_config(self):
        argv = self.launch("openhands", {"llm_config": "configs/paper model.json",
                                         "workers": 1, "output": "outputs/repeat 0"})
        self.assertEqual(argv[:4], ["uv", "run", "swebench-infer", "configs/paper model.json"])
        self.assertFalse(any("temperature" in item or "top_p" in item or "top-p" in item
                             for item in argv))
        with CONFIG.open("rb") as stream:
            systems = tomllib.load(stream)["systems"]
        self.assertEqual({name for name, value in systems.items() if "argv" in value},
                         {"swe-agent", "mini-swe-agent", "openhands"})


if __name__ == "__main__":
    unittest.main()
