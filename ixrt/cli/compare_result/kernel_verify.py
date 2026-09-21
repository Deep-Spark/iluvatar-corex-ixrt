# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#
import json
import os
from typing import List

from tabulate import tabulate

from .formatting import green, red, yellow


def print_kernel_verify_report(report_path: str) -> bool:
    """Load the IXRT_VERIFY_KERNEL json report and print a Phase-1 style table.

    Returns True if no WRONG/ERROR entries exist.
    """
    if not report_path or not os.path.exists(report_path):
        print(f"[verify_kernel] report not found: {report_path}")
        return False

    with open(report_path, "r") as f:
        rows = json.load(f)

    if not rows:
        print("[verify_kernel] empty report (no layers verified)")
        return True

    table = []
    ok = True
    for item in rows:
        status = item.get("status", "ERROR")
        if status in ("WRONG", "ERROR"):
            ok = False
            status_s = red(status)
        elif status == "SKIPPED":
            status_s = yellow(status)
        else:
            status_s = green(status)

        config = item.get("config", {})
        if isinstance(config, dict):
            config_s = json.dumps(config, separators=(",", ":"))
        else:
            config_s = str(config)

        # nlohmann::json serializes non-finite floats (inf/nan) as null, so metrics
        # may come back as None; coerce to a float for safe formatting.
        def _num(key):
            v = item.get(key, 0)
            try:
                return float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        metrics = (
            f"cos={_num('cosine_sim'):.6f}\n"
            f"diff_max={_num('diff_max')}\n"
            f"diff_rel_avg={_num('diff_rel_avg')}\n"
            f"{item.get('message', '')}"
        )
        table.append(
            [
                status_s,
                f"{item.get('op_name', '')}\nkernel: {item.get('fn_signature', '')}\nconfig: {config_s}",
                metrics,
            ]
        )

    print(
        tabulate(
            table,
            headers=["Result", "Operator / Kernel / Config", "vs Naive Ref"],
            tablefmt="grid",
        )
    )
    wrong = sum(1 for r in rows if r.get("status") == "WRONG")
    skipped = sum(1 for r in rows if r.get("status") == "SKIPPED")
    error = sum(1 for r in rows if r.get("status") == "ERROR")
    right = sum(1 for r in rows if r.get("status") == "RIGHT")
    print(
        f"[verify_kernel] summary: RIGHT={right} WRONG={wrong} SKIPPED={skipped} ERROR={error}"
    )
    # Emit the signature of every SKIPPED op counted in the summary above. These
    # lines come from the exact report the summary is computed from (the main
    # inference pass), so downstream tooling can assert the permitted skip set
    # without being polluted by auxiliary passes (e.g. one-off constant-weight
    # reformats) whose console lines are not part of this summarized run.
    for r in rows:
        if r.get("status") == "SKIPPED":
            print(
                f"[verify_kernel] summary_skip: {r.get('op_name','')} | "
                f"kernel: {r.get('fn_signature','')} | {r.get('message','')}"
            )
    return ok and error == 0
