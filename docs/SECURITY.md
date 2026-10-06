# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities privately via [GitHub Security Advisories](https://github.com/sylenshi/long-horizon-swe-agent/security/advisories/new)
for this repository, or contact the maintainer ([@sylenshi](https://github.com/sylenshi)) directly.

Please do not open a public issue for security reports.

## Scope

- This project inherits the upstream [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) threat model: the agent executes arbitrary bash commands in a sandbox you configure. You are responsible for the sandbox isolation (e.g. docker, bubblewrap, or a disposable VM).
- Reports about issues in upstream code that also affect [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) are welcome here; we will coordinate with upstream where appropriate.
- New code in this fork is concentrated in `src/minisweagent/context/` (context compaction, accounting, event logging) and the `DefaultAgent` view-projection hooks.
