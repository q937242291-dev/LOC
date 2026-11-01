# Sampling settings in public agents

The table lists settings from fixed public source versions, checked on 2026-10-31. `Not set` means the configuration does not explicitly supply the field. `None` means the configuration type defaults to an empty value. Requests also pass through the model adapter and selected provider.

| System and version | temperature | top_p | LLM seed | Official source |
|---|---|---|---|---|
| SWE-agent `v1.1.0`, model type used by the default configuration | `0.0` | `1.0` | Not set; `completion_kwargs` defaults to `{}` and accepts extra request parameters | [GenericAPIModelConfig and request forwarding](https://github.com/SWE-agent/SWE-agent/blob/v1.1.0/sweagent/agent/models.py#L62), [default configuration](https://github.com/SWE-agent/SWE-agent/blob/v1.1.0/config/default.yaml) |
| mini-swe-agent `v2.4.6`, SWE-bench configuration with the LiteLLM model | Not set | Not set | Not set; parameters pass through `model.model_kwargs` | [SWE-bench configuration](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/config/benchmarks/swebench.yaml#L165), [LiteLLM model](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/models/litellm_model.py#L23) |
| Agentless `v1.5.0`, official SWE-bench workflow | `0` and `0.8` by stage; failed localization retries use `1.0`; see below | Not set by the request constructor | Not set by the request constructor | [Official experiment commands](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/README_swebench.md), [API request construction](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/agentless/util/api_requests.py#L20) |
| OpenHands Software Agent SDK `v1.59.0`, `LLM` type defaults | `None`, using the provider default | `None`, using the provider default | `None`; the configuration object provides a `seed` field | [LLM sampling fields](https://github.com/OpenHands/software-agent-sdk/blob/v1.59.0/openhands-sdk/openhands/sdk/llm/llm.py#L394), [seed field](https://github.com/OpenHands/software-agent-sdk/blob/v1.59.0/openhands-sdk/openhands/sdk/llm/llm.py#L603) |
| Live-SWE-agent `8d7dd8634580d1e09320b4c27d70380bc9ae74a8`, `livesweagent.yaml` | `0.0` | Not set | Not set | [Configuration at the fixed commit](https://github.com/OpenAutoCoder/live-swe-agent/blob/8d7dd8634580d1e09320b4c27d70380bc9ae74a8/config/livesweagent.yaml#L182) |

The OpenHands row describes type defaults in the specified SDK version. With [OpenHands benchmarks](https://github.com/OpenHands/benchmarks/blob/main/benchmarks/swebench/README.md), the entry point reads the supplied LLM JSON configuration and uses that checkout's `vendor/software-agent-sdk`. Pin the benchmark repository and SDK versions together with the model configuration. Select settings for older OpenHands releases or specific model experiments from their corresponding versions.

## Agentless settings by stage

| Stage | Official setting | Official source |
|---|---|---|
| Relevant file localization and irrelevant directory filtering | `temperature=0` | [FL.py](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/agentless/fl/FL.py#L219) |
| Relevant function, class, and other element localization | Initially `0`; subsequent attempts use `1.0` when no valid location is found | [localize.py](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/agentless/fl/localize.py#L159) |
| Fine-grained edit location localization | Official experiment command: `0.8`, 4 samples; uses `1.0` when no valid location is found. The CLI `--temperature` default is `0.0` | [Experiment command](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/README_swebench.md#L121), [localization retries](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/agentless/fl/localize.py#L255), [CLI default](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/agentless/fl/localize.py#L508) |
| Patch generation | First sample in each group: `0`; remaining samples: `0.8`. The official command uses 10 samples per group across 4 groups | [repair.py](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/agentless/repair/repair.py#L337), [experiment command](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/README_swebench.md#L168) |
| Regression test selection | `0` | [select_regression_tests.py](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/agentless/test/select_regression_tests.py#L86) |
| Reproduction test generation | First sample: `0`; remaining samples: `0.8`. The official experiment command uses 40 samples | [generate_reproduction_tests.py](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/agentless/test/generate_reproduction_tests.py#L129), [experiment command](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/README_swebench.md#L234) |

The Agentless decoder base class defaults to `temperature=0.8`. The stages above explicitly override that value; the calling stage determines the requested temperature. [Decoder and request calls](https://github.com/OpenAutoCoder/Agentless/blob/v1.5.0/agentless/util/model.py#L6)

## Paper and upstream protocols

The paper protocol requests uniform `temperature=0.7` and `top_p=1.0`. Record these experiment settings separately from the defaults of each upstream version above. To compare original upstream configurations, run a separate group that retains their stage temperatures and sample counts. Use separate output directories, system configuration identifiers, and aggregate results.

| Run | Configuration |
|---|---|
| SWE-agent with uniform sampling | The project's Lite template includes `--agent.model.temperature 0.7 --agent.model.top_p 1.0`. These fields belong to the model configuration above; batch mode accepts all `--agent` configuration arguments. A provider seed can be requested through this version's `completion_kwargs`. [CLI configuration](https://swe-agent.com/latest/usage/cl_tutorial/#configuration-files), [batch arguments](https://swe-agent.com/latest/usage/batch_mode/#a-first-example-swe-bench). |
| mini-swe-agent with uniform sampling | The project's Lite template includes `-c swebench.yaml -c model.model_kwargs.temperature=0.7 -c model.model_kwargs.top_p=1.0`. Setting `-c` replaces the default configuration selection, so load the base configuration explicitly before overriding sampling fields. [Configuration merging](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/run/benchmarks/swebench.py#L32) |
| Agentless with uniform sampling | Adjust the temperatures and retry branches in each stage, and set `top_p` in the API request constructor. Adding only `--temperature 0.7` to localization changes fine-grained localization; repair and test generation keep their source settings. The stage table lists the relevant source locations. |
| OpenHands with uniform sampling | Add `"temperature": 0.7, "top_p": 1.0` to the runtime LLM JSON object passed as `llm_config`. Set its `seed` field when requesting a provider seed. The official launcher loads this JSON from a positional argument and provides no temperature/top-p CLI overrides. [Argument definitions](https://github.com/OpenHands/benchmarks/blob/main/benchmarks/utils/args_parser.py#L14), [configuration loading](https://github.com/OpenHands/benchmarks/blob/main/benchmarks/utils/llm_config.py#L7), [SDK fields and forwarding](https://github.com/OpenHands/software-agent-sdk/blob/v1.59.0/openhands-sdk/openhands/sdk/llm/llm.py#L2363). |

Configure sampling fields according to the provider and model interface. The two-parameter overrides above apply to endpoints that accept both `temperature` and `top_p`. Record requested configuration values and fields actually sent separately, along with the configuration version and endpoint model identifier.

For Sonnet 4.5, use `temperature=0.7` and configure the upstream model adapter to omit `top_p` from the request. Adjust the two-parameter templates before using them with Sonnet 4.5. The SWE-agent model type itself defaults to `top_p=1.0`; removing only the command-line override retains that default, so also adjust its configuration or request forwarding. Anthropic's official migration instructions require at most one of `temperature` and `top_p` for Claude 4.x and state that supplying both causes an error. [Official migration instructions](https://github.com/anthropics/skills/blob/main/skills/claude-api/shared/model-migration.md#L342) Record `top_p` as omitted and the provider default as `0.99`. The Messages API introduced this default on 2025-05-22; omitting the field does not request `top_p=1.0`. [Official release notes](https://platform.claude.com/docs/en/release-notes/overview#may-22-2025)

## Five repeats and seeds

`repeat` identifies five independent attempts. A logical seed controls local random operations; a provider seed is an API request field. The mini-swe-agent `v2.4.6` `--shuffle` option uses `random.seed(42)` only to control task order. [Task ordering implementation](https://github.com/SWE-agent/mini-swe-agent/blob/v2.4.6/src/minisweagent/run/benchmarks/swebench.py#L167)

SWE-agent's extra parameters, mini's `model_kwargs`, and OpenHands' `seed` field provide places to pass a provider seed. Send it according to the selected model endpoint's support. Record each attempt's logical seed, provider seed request, and task order seed separately.
