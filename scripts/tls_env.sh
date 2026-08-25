# shellcheck shell=bash
#
# Point TLS-using tools at the system CA bundle. Source this, do not execute it.
#
#   source "$REPO_DIR/scripts/tls_env.sh"
#
# WHY
# ---
# eX3 sits behind a TLS-inspecting proxy: connections are re-signed by a local
# CA that lives in the system trust store. curl and git therefore work, because
# they use the system bundle. Tools that ship their own roots do not:
#
#   uv       bundles webpki roots  -> "invalid peer certificate: UnknownIssuer"
#   requests bundles certifi       -> SSLCertVerificationError
#
# Both honour environment variables pointing at a bundle, so we find the system
# one and export it under every name the relevant tools look for.
#
# This trusts whatever the host already trusts — it does not disable
# verification. If no bundle is found we leave everything alone rather than
# weakening TLS; the caller will get the original error, which is the honest
# outcome.

_bris_find_ca_bundle() {
  local c
  for c in \
    /etc/ssl/certs/ca-certificates.crt \
    /etc/pki/tls/certs/ca-bundle.crt \
    /etc/ssl/ca-bundle.pem \
    /etc/ssl/cert.pem
  do
    [[ -r "$c" ]] && { echo "$c"; return 0; }
  done
  return 1
}

if BRIS_CA_BUNDLE="$(_bris_find_ca_bundle)"; then
  export BRIS_CA_BUNDLE
  # OpenSSL, and anything built on it
  export SSL_CERT_FILE="${SSL_CERT_FILE:-$BRIS_CA_BUNDLE}"
  export SSL_CERT_DIR="${SSL_CERT_DIR:-$(dirname "$BRIS_CA_BUNDLE")}"
  # python-requests / huggingface_hub
  export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$BRIS_CA_BUNDLE}"
  # curl
  export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-$BRIS_CA_BUNDLE}"
  # uv: use the platform trust store instead of its bundled webpki roots.
  # The flag has been spelled --native-tls and --system-certs across versions;
  # both env vars are set, and uv ignores the one it does not know.
  export UV_NATIVE_TLS="${UV_NATIVE_TLS:-1}"
  export UV_SYSTEM_CERTS="${UV_SYSTEM_CERTS:-1}"
else
  echo "WARNING: no system CA bundle found; TLS-inspecting proxies will break uv/pip" >&2
fi
