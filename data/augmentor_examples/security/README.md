# Security Augmentor Library

Domain: `security`. 12 YAML files, 21 examples, covering authorized
security-testing workflows (pentesting, physical access testing, red
team support, security research).

## What this library is for

When ulcagent is asked to build a tool for an authorized security
engagement, these examples surface via the augmentor retrieval system
and give the model strong scaffolding for common patterns. The 14B
already knows scapy/nmap/nfcpy/bleak/etc. at a baseline level; these
examples lock in the patterns Jordan uses across engagements
(error handling, output formatting, integration with the reporting
pipeline) so the model doesn't reinvent the wiring each time.

## Categories

| Category | Files | Topics |
|---|---|---|
| Network | 5 | ARP scan, port scan + banner, packet capture + replay, DNS enum + AXFR, nmap wrapper |
| Physical access | 3 | Proxmark3 wrapper + LF clone, NFC MIFARE dump + diff, Wiegand serial listener |
| Wireless | 2 | iwlist scan + rogue AP detect, BLE GATT enum + passive ad logging |
| Reporting | 2 | findings → markdown + CSV, pcap → analysis summary |

## Authorization expectations

These examples assume operation within an **authorized engagement context**:

- A signed scope-of-work (SOW) defining target systems, networks, and
  testing windows
- Client-side approvals for any actions outside passive enumeration
  (badge cloning, deauth testing, exploitation attempts)
- A per-engagement workspace separate from any other client's data

Several examples include explicit engagement notes (e.g., "passive
monitoring may still be out of scope on medical/OT networks"; "badge
cloning requires both badge owner AND system owner approval"). Read
those notes — they're the difference between a clean engagement and
a legal exposure.

## What's intentionally NOT in here

- **Specific exploits / 0-days.** The 14B's training cutoff makes any
  CVE-specific code stale or wrong; better to fetch live exploit-db
  entries via `--web` per engagement.
- **Credential brute-force tools.** Hashcat / hydra wrappers are easy
  to write directly; they're not subtle enough to need an augmentor.
  Also, brute-force is the loudest possible technique and rarely the
  right tool for a real engagement.
- **Automated exploitation chains.** Anything that goes from "scan"
  to "shell" without human judgment in between. The model isn't good
  enough to make those decisions yet, and engagements should have
  human gates regardless.
- **Defensive evasion / anti-forensics.** Not in scope for white-hat
  testing — clients want findings + remediation, not evidence of how
  to hide post-compromise.

## How retrieval surfaces these

Two paths:

1. **Embedding similarity.** When a goal mentions scapy, nmap, RFID,
   Proxmark, BLE, MIFARE, Wiegand, etc., the augmentor router computes
   query embeddings and pulls the most-similar examples.
2. **Agentic keyword gate.** `_LARGE_MODE_AGENTIC_KEYWORDS` in
   `engine/augmentors.py` includes security-domain keywords (pentest,
   security testing, sniffer, scanner, etc.) so retrieval fires for
   the 14B even when augmentors would otherwise be skipped on Python
   queries.

## Extending the library

Add a new YAML file under this directory following the existing format:

```yaml
domain: security
category: <descriptive_category_name>
examples:
- query: |
    <one user goal>
  solution: |
    <code + commentary>
  tags: [<tag1>, <tag2>, ...]
  difficulty: easy | medium | hard
```

After adding, add the new category's distinguishing keywords to
`_LARGE_MODE_AGENTIC_KEYWORDS` in `engine/augmentors.py` if the
existing keywords don't already cover them.

## Practical reminders

- **The model is a junior engineer.** Read what it writes before running
  it against real targets. A buggy ARP scanner is annoying; a buggy
  deauth tool can brick old IoT devices.
- **Test in your lab first.** Every engagement-bound tool should run
  in a controlled lab against your own gear before you point it at a
  client.
- **Log everything.** Pentest deliverables ARE the logs. Build logging
  in from the first version of any tool, not as an afterthought.
- **Respect the engagement scope.** The augmentors don't enforce scope —
  that's on you. Hard-code scope allowlists per engagement and fail
  closed when a target isn't in the allowlist.
