<!-- SPDX-License-Identifier: MPL-2.0 -->
# Security reports

This is a developer alpha; there is no support or security-response SLA.
Never put customer documents, credentials or workspace databases in public issues.
Use a fictional reproducer for public bug reports. For a sensitive vulnerability,
use [GitHub private vulnerability reporting](https://github.com/fhajjej-ship-it/bidtrace-core/security/advisories/new).

The supported full-analysis setup is local MLX on Apple Silicon. Keep the model
service bound to loopback. The core CLI accepts only a credential-free loopback
HTTP service and does not require hosted inference or application dotenv files.
Explicit model preparation downloads pinned public artifacts; analysis runs
with offline model settings. Dependencies and models retain separate rights.
