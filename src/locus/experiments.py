"""Controller configurations for the paper's selection and sensitivity studies."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product


@dataclass(frozen=True)
class Case:
    name: str
    settings: object


def cases(study, base):
    if study == "selection":
        return [Case(name,replace(base,selection=policy,packet_tokens=cap))
                for name,policy,cap in (
                    ("EditOnly","edit_only",16000),
                    ("StaticTopK","static_topk",16000),("ExactDedup","exact_dedup",16000),
                    ("LOCUS","coverage",16000))]
    if study == "components":
        changes = (
            ("LOCUS",{}),
            ("Without-role-grounding",dict(role_grounding=False)),
            ("Without-alternatives",dict(alternatives=False)),
            ("Without-gate",dict(gate_enabled=False)),
            ("Without-memory",dict(memory=False)),
            ("Without-packet",dict(selection="broad",packet_tokens=32000)),
            ("Without-packet-memory",dict(selection="broad",packet_tokens=32000,memory=False)),
        )
        return [Case(name,replace(base,**change)) for name,change in changes]
    if study == "allocation":
        return [Case("LOCUS" if a == "adaptive" else "Uniform-"+a,replace(base,allocation=a))
                for a in ("adaptive","L","M","H")]
    if study == "width":
        return [Case("LOCUS-K"+str(k),replace(base,frontier_size=k))
                for k in (1,3,5,8,12,18)]
    if study == "capacity":
        # All four evidence conditions remain active; the window controls the
        # persistence required before handing off to repair.
        return [Case(f"LOCUS-{cap//1000}k-{label}",
                     replace(base,packet_tokens=cap,stability_window=window))
                for cap,(label,window) in product(
                    (4000,8000,16000,24000),
                    (("Aggressive",1),("Balanced",2),("Strict",3)))]
    raise ValueError("Unknown experiment study")


def main_matrix():
    systems = ("SWE-agent","Agentless","OpenHands","Live-SWE-agent","CCA",
               "Claude-Code","Codex","mini-swe-agent","LOCUS")
    return [dict(benchmark=b,model=m,system=s,repeats=5,tasks=n,attempts=5*n)
            for (b,n),m,s in product(
                (("lite",300),("pro-v1",731),("pro-v2",642)),
                ("DeepSeek-V4-Pro","Claude-Sonnet-4.5","GPT-5.2"),systems)]

