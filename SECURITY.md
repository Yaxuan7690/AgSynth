# Security policy

## Credentials

Store credentials only in local environment variables or a local `.env` file. The repository ignores `.env`. Revoke and rotate any credential that is accidentally committed.

## Generated renderer code

AgSynth executes model-generated Matplotlib code. The runner applies AST checks, an import allowlist, a fixed output destination, a timeout, and a sanitized subprocess environment. These controls reduce accidental misuse but do not constitute a hardened sandbox against a determined adversary.

Run generation in a dedicated virtual machine, container, or disposable environment. Do not run unreviewed generated code on systems containing credentials, proprietary datasets, or sensitive files.

## Reporting

Do not publish credentials or private input data in issue reports. Contact the repository maintainer through the private channel specified by the project owner for security-sensitive reports.
