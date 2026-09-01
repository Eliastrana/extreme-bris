"""One place for the eX3 TLS problem, so it stops being rediscovered.

eX3 intercepts HTTPS with its own CA. Every client that verifies certificates
therefore needs the system bundle pointed at explicitly, or it fails with

    SSLCertVerificationError: self-signed certificate in certificate chain

which reads like the remote service is broken. scripts/env.sh sets this up, but
scripts get run from shells where it was never sourced, and then the traceback
sends you looking in the wrong place.

This has now cost time three times: the OPeNDAP probes, where it was misread as
a protocol problem for several rounds; the CDS fetch; and the Frost fetch, where
the guard existed in one script and had not been carried into the next one
written. Hence a module rather than a copied function.

Call ensure_ca_bundle() before importing or constructing any HTTP client - most
of them read the environment once, at import or construction.
"""

from __future__ import annotations

import os
import sys

# Same list scripts/tls_env.sh searches, in the same order.
CA_CANDIDATES = (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    "/etc/ssl/cert.pem",
)


def ensure_ca_bundle(quiet: bool = False) -> str | None:
    """Point HTTPS clients at the system CA bundle. Returns the path, or None."""
    for var in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        existing = os.environ.get(var)
        if existing:
            return existing

    for cand in CA_CANDIDATES:
        if os.access(cand, os.R_OK):
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cand)
            os.environ.setdefault("SSL_CERT_FILE", cand)
            os.environ.setdefault("CURL_CA_BUNDLE", cand)
            if not quiet:
                print(f"note: no CA bundle in the environment; using {cand}")
                print("      (sourcing scripts/env.sh does this properly)\n")
            return cand

    print("WARNING: no CA bundle found and none configured. On eX3 the request",
          file=sys.stderr)
    print("         will fail TLS verification. Run: source scripts/env.sh\n",
          file=sys.stderr)
    return None


def explain(exc: BaseException, host: str) -> str | None:
    """A one-line diagnosis for a TLS failure, or None if it is something else."""
    text = f"{type(exc).__name__}: {exc}"
    if "CERTIFICATE_VERIFY_FAILED" not in text and "SSLError" not in text:
        return None
    return (f"\nTLS verification failed against {host}.\n"
            "This is eX3 intercepting HTTPS, not a problem with the service or\n"
            "your credentials. Run:\n\n"
            "  source scripts/env.sh\n\n"
            "and try again from the same shell.\n")
