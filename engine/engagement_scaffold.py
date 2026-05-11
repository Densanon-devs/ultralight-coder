"""Engagement workspace scaffolding for security testing engagements.

Creates a standard directory layout and template files inside a new
engagement directory. Designed to run via `ulcagent --new-engagement <name>`,
but also usable programmatically.

Layout created:
    <name>/
        .ulcagent              # Engagement-specific agent instructions
        README.md              # Engagement overview (template, fill in)
        scope/
            sow.md             # Statement of Work template
            targets.txt        # Allowed target hosts (one per line)
            out_of_scope.txt   # Explicit out-of-scope targets
        evidence/              # Captured evidence (pcaps, screenshots)
            .gitkeep
        findings/
            findings.json      # Structured findings DB ([] initially)
            ARCHIVE/           # Per-day archived snapshots
                .gitkeep
        tools/                 # Engagement-specific scripts the agent writes
            .gitkeep
        audit/                 # Per-call audit log (auto-populated by AuditLog)
            .gitkeep
        report/
            README.md          # Notes for report assembly
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path


ULCAGENT_TEMPLATE = """\
# {client} engagement — agent instructions

This is an AUTHORIZED security testing engagement. Standard rules:

1. **Scope is enforced by the operator.** Before testing any target, verify
   it appears in scope/targets.txt. If it's listed in scope/out_of_scope.txt
   under any circumstances, do NOT touch it. Both files are authoritative.

2. **Write tools to tools/.** Any script you generate goes there. Don't
   pollute the workspace root.

3. **Write findings to findings/findings.json.** Append to the existing
   JSON array — don't overwrite. Each finding is a dict:
   {{title, severity, host, description, remediation, evidence, first_seen}}.

4. **Evidence goes to evidence/.** Pcaps, screenshots, command output,
   captured credentials, scan results. One subdir per category if it
   helps organization.

5. **The audit/ directory captures every tool call automatically.** Don't
   write there manually.

6. **For the final deliverable**, build it under report/. The findings
   reporter (data/augmentor_examples/security/reporting_findings_to_markdown.yaml)
   reads from findings/findings.json — use it.

7. **Privacy and air-gap.** Run --web only when looking up public docs
   (CVE detail, vendor docs, exploit-db). Do NOT search for queries that
   reveal client identity or specific findings. The model's --web calls
   are logged to audit/ — they're discoverable.

[aliases]
/scope = Read scope/targets.txt and scope/out_of_scope.txt; tell me which targets are in scope.
/findings = Read findings/findings.json and summarize what's been logged so far.
/render-report = Build report/{client}-report.md from findings/findings.json using the standard severity-grouped Markdown template.
"""


README_TEMPLATE = """\
# {client} — Pentest Engagement

**Client:** {client}
**Tester:** Densanon Security
**Engagement window:** {start_date} to TBD
**Statement of Work:** see [`scope/sow.md`](scope/sow.md)

## Workspace layout

| Path | Purpose |
|---|---|
| `scope/` | SOW + target allowlist + out-of-scope list |
| `evidence/` | Pcaps, screenshots, command output |
| `findings/findings.json` | Structured findings DB |
| `tools/` | Engagement-specific scripts |
| `audit/` | Per-call audit log (auto-populated) |
| `report/` | Final client deliverable |

## Quick commands inside ulcagent

- `/scope` — show what's in/out of scope
- `/findings` — summary of logged findings
- `/render-report` — build the final markdown report

## Workflow

1. Confirm scope (`/scope`).
2. Run scoped tools from `tools/` (or have ulcagent build them).
3. Append findings to `findings/findings.json` as they surface.
4. Capture evidence to `evidence/`.
5. When complete, render the report (`/render-report`).
"""


SOW_TEMPLATE = """\
# Statement of Work — {client}

> **TODO**: replace placeholders before engagement starts.

## Engagement details

- **Client:** {client}
- **Engagement type:** [external pentest | internal pentest | physical security | wireless | webapp | other]
- **Engagement window:** {start_date} to YYYY-MM-DD
- **Tester:** Densanon Security (Jordan)
- **Client point-of-contact:** TODO
- **Authorization document:** TODO (link or attach)

## In-scope targets

See `targets.txt` for the authoritative list of hostnames/IPs/CIDRs.

## Out-of-scope

See `out_of_scope.txt` for the authoritative list of exclusions.

## Authorized techniques

- [ ] Passive reconnaissance (DNS enum, public OSINT, banner grabbing)
- [ ] Active scanning (nmap, port scans against targets)
- [ ] Service-version probing
- [ ] Vulnerability scanning (specify tools in note)
- [ ] Manual exploitation (with stop conditions, see below)
- [ ] Credential testing (with provided test accounts only)
- [ ] Social engineering
- [ ] Physical access testing (badge cloning, tailgating, etc.)
- [ ] Wireless audit
- [ ] Web application testing (with stop conditions, see below)

## Stop conditions

If any of the following occur, STOP immediately, document, and notify the
client POC before continuing:
- Unintended denial of service against any target
- Access to data outside the agreed scope (PII, financial, medical)
- Discovery of an actively-exploited vulnerability (existing breach indicator)
- Any finding rated CVSS 9.0+ (Critical)

## Reporting

- Format: Markdown (rendered to PDF for delivery)
- Severity scale: CVSS v3.1 (Critical / High / Medium / Low / Informational)
- Delivery: TODO (email, secure drop, etc.)
- Confidentiality: All findings + evidence remain in this engagement
  workspace. Nothing leaves without client written approval.
"""


REPORT_README = """\
# Report assembly

Final deliverable goes here. Suggested filename: `<client>-pentest-YYYY-MM-DD.md`.

## Build via ulcagent

```
/render-report
```

This wraps the standard severity-grouped Markdown template (see the
augmentor library `data/augmentor_examples/security/reporting_findings_to_markdown.yaml`)
against `findings/findings.json`.

## Manual touches before delivery

- [ ] Executive summary (2-3 paragraphs, plain English, business-impact focused)
- [ ] Verify all "TODO" markers in the SOW are filled in
- [ ] Confirm all evidence references are valid (links work, files exist)
- [ ] Strip any internal-only notes from findings descriptions
- [ ] Run a final spell/grammar pass
- [ ] Convert to PDF if required by client
"""


TARGETS_TXT_TEMPLATE = """\
# Authorized targets — one per line.
# Hostnames or IPs or CIDR ranges. Comments allowed via #.
# Example:
#   client.example.com
#   192.0.2.0/24
#   10.10.0.0/16
#
# CONFIRM THIS LIST AGAINST THE SOW BEFORE TESTING ANY TARGET.

"""


OUT_OF_SCOPE_TXT_TEMPLATE = """\
# Explicitly out-of-scope — one per line.
# Hostnames, IPs, or CIDR ranges that MUST NOT be touched even if they
# appear adjacent to in-scope assets.
# Example:
#   admin.client.example.com   # production admin panel — explicitly excluded
#   10.20.0.0/16               # finance segment
#
# These take precedence over targets.txt — if a target appears in both,
# treat it as OUT OF SCOPE.

"""


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower())
    return re.sub(r"-{2,}", "-", s).strip("-")


def create_engagement(client: str, parent_dir: Path | str = ".",
                      start_date: str | None = None) -> Path:
    """Create the engagement workspace under `<parent_dir>/<slug-of-client>/`.

    Returns the absolute path to the created engagement directory.
    Raises FileExistsError if the target directory already exists.
    """
    if not client or not client.strip():
        raise ValueError("client name is required")
    slug = _slugify(client)
    if not slug:
        raise ValueError(f"could not derive slug from client name {client!r}")
    parent = Path(parent_dir).resolve()
    eng = parent / slug
    if eng.exists():
        raise FileExistsError(f"{eng} already exists — pick a different name or delete it first")

    start = start_date or date.today().isoformat()

    # Build directory tree
    eng.mkdir(parents=True)
    for sub in ("scope", "evidence", "findings", "findings/ARCHIVE",
                "tools", "audit", "report"):
        (eng / sub).mkdir(parents=True, exist_ok=True)
    # .gitkeep for empty dirs
    for sub in ("evidence", "findings/ARCHIVE", "tools", "audit"):
        (eng / sub / ".gitkeep").touch()

    # Template files
    (eng / ".ulcagent").write_text(
        ULCAGENT_TEMPLATE.format(client=client), encoding="utf-8")
    (eng / "README.md").write_text(
        README_TEMPLATE.format(client=client, start_date=start), encoding="utf-8")
    (eng / "scope" / "sow.md").write_text(
        SOW_TEMPLATE.format(client=client, start_date=start), encoding="utf-8")
    (eng / "scope" / "targets.txt").write_text(TARGETS_TXT_TEMPLATE, encoding="utf-8")
    (eng / "scope" / "out_of_scope.txt").write_text(OUT_OF_SCOPE_TXT_TEMPLATE, encoding="utf-8")
    (eng / "findings" / "findings.json").write_text("[]\n", encoding="utf-8")
    (eng / "report" / "README.md").write_text(REPORT_README, encoding="utf-8")

    return eng
