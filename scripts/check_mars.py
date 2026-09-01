#!/usr/bin/env python3
"""Probe MARS access with the smallest retrieval that still proves anything.

    source scripts/env.sh          # NOT optional - see stage 0
    python3 scripts/check_mars.py

The point is to find out whether `class: od` is actually open to us BEFORE
submitting a request for 89 fields. A request the licence does not cover is not
refused immediately: it can sit queued and fail an hour later, so the cheap
probe is worth the two minutes.

Four stages, cheapest first, each one gating the next:

  0. environment    - client importable, credentials present, TLS sane
  1. identity       - who does ECMWF think we are (no queue, no retrieval)
  2. retrieval      - ONE parameter, ONE level, ONE date, ONE time
  3. grid           - does N320 come back as 542,080 reduced-Gaussian points

Stage 3 is not paranoia. `grid: N320` is a server-side interpolation - recent
operational cycles are native O1280 octahedral - and a request that silently
returns a regular lat/lon grid will build a dataset that does not match the
checkpoint's graph. The number to see is 542,080: subtract the 5,480 points the
cutout drops under the LAM and you get 536,600, which is what we read out of
the checkpoint.

If `od` is refused, the script retries the identical request against `ea`
(ERA5, which we know works). That distinguishes "no licence for od" from
"the client, credentials or network is broken" - two failures that otherwise
look identical.

Exit codes:  0 od works | 1 od refused, harness fine | 2 setup broken | 3 timeout
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

N320_POINTS = 542_080
CUTOUT_DROPS = 5_480

OK, BAD, WARN, INFO = "  ok  ", " FAIL ", " warn ", "      "


def say(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


# --- stage 0 ----------------------------------------------------------------

def stage_env() -> tuple[bool, object]:
    say(INFO, "--- stage 0: environment")

    try:
        from ecmwfapi import ECMWFService  # noqa: F401
        import ecmwfapi
    except ImportError:
        say(BAD, "ecmwf-api-client not importable.")
        say(INFO, "    pip install ecmwf-api-client   (in the DATASET env, not")
        say(INFO, "    the inference lockfile - see docs/INPUTS.md)")
        return False, None
    say(OK, f"ecmwf-api-client {getattr(ecmwfapi, '__version__', 'unknown')}")

    # Credentials: file or environment. Never print the key itself.
    rc = Path.home() / ".ecmwfapirc"
    env_key = os.environ.get("ECMWF_API_KEY")
    if rc.exists():
        try:
            import json
            conf = json.loads(rc.read_text())
            missing = [k for k in ("url", "key", "email") if not conf.get(k)]
            if missing:
                say(BAD, f"{rc} is missing: {', '.join(missing)}")
                return False, None
            say(OK, f"{rc} has url, key, email  (url={conf['url']})")
            mode = int(rc.stat().st_mode) & 0o077
            if mode:
                say(WARN, f"{rc} is readable by others (mode {oct(rc.stat().st_mode)[-3:]}).")
                say(INFO, f"    eX3 is shared: chmod 600 {rc}")
        except Exception as exc:
            say(BAD, f"{rc} is not valid JSON: {exc}")
            return False, None
    elif env_key:
        if not os.environ.get("ECMWF_API_EMAIL"):
            say(BAD, "ECMWF_API_KEY set but ECMWF_API_EMAIL is not.")
            return False, None
        say(OK, "credentials from ECMWF_API_* environment variables")
    else:
        say(BAD, f"no credentials: neither {rc} nor ECMWF_API_KEY.")
        say(INFO, "    get them from https://api.ecmwf.int/v1/key/")
        return False, None

    # The lesson from the OPeNDAP probes: eX3 intercepts TLS, and a probe run
    # without the CA bundle fails in a way that looks like a protocol problem.
    # Do not let that happen twice.
    if not (os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")):
        say(WARN, "no SSL_CERT_FILE / REQUESTS_CA_BUNDLE in the environment.")
        say(INFO, "    On eX3 that means TLS interception will look like a")
        say(INFO, "    certificate error and you will misread it as a network")
        say(INFO, "    fault.  source scripts/env.sh  and run this again.")
    else:
        say(OK, "CA bundle configured for HTTPS clients")

    from ecmwfapi import ECMWFService
    return True, ECMWFService


# --- stage 1 ----------------------------------------------------------------

def stage_identity(ECMWFService) -> bool:
    say(INFO, "--- stage 1: identity")
    try:
        svc = ECMWFService("mars")
        # The client resolves credentials on construction; ask it who we are.
        # This does not queue anything.
        who = None
        for attr in ("email", "user", "_email"):
            who = getattr(svc, attr, None)
            if who:
                break
        # Constructing the service resolves credentials locally. It says
        # nothing about authorisation - that only shows up on the first call.
        say(OK, f"credentials resolved{f' as {who}' if who else ''}")
        return True
    except Exception as exc:
        say(BAD, f"could not construct the MARS service: {exc}")
        return False


# --- stage 2 ----------------------------------------------------------------

def tiny_request(cls: str, date: str, time_: str, levtype: str, grid: str) -> dict:
    """One parameter, one level, one date, one time. Nothing smaller proves as much."""
    req = {
        "class": cls,
        "stream": "oper",
        "type": "an",
        "expver": "1" if cls == "od" else "0001",
        "date": date,
        "time": time_,
        "grid": grid,
        "levtype": levtype,
        "param": "2t" if levtype == "sfc" else "t",
    }
    if levtype == "pl":
        req["levelist"] = "500"
    return req


def retrieve(ECMWFService, req: dict, target: Path, timeout: int) -> tuple[str, str]:
    """Returns (status, detail). status in {ok, refused, timeout, error, unreachable}.

    The client retries network errors internally, on a 60s cycle, and prints
    rather than raises while it does. Left alone, a host we cannot reach at all
    looks exactly like a request sitting in the MARS queue - so capture its log
    and read the difference out of it.
    """
    result: dict = {}
    chatter: list[str] = []

    def log(msg: str) -> None:
        text = str(msg).strip()
        if text:
            chatter.append(text)
            say(INFO, "    | " + text)

    def worker() -> None:
        try:
            ECMWFService("mars", log=log).execute(req, str(target))
            result["status"] = "ok"
        except Exception as exc:                      # noqa: BLE001
            text = f"{type(exc).__name__}: {exc}"
            low = text.lower()
            # "has no access to services/mars" is the whole service being off,
            # not this class being closed. The two need different advice, and
            # retrying another class against the same service proves nothing.
            if "no access to services" in low:
                result["status"] = "no_service"
            else:
                denied = ("not authoris", "not author", "access denied", "forbidden",
                          "no access", "permission", "licence", "license", "401", "403")
                result["status"] = "refused" if any(d in low for d in denied) else "error"
            result["detail"] = text

    t = threading.Thread(target=worker, daemon=True)
    started = time.time()
    t.start()
    t.join(timeout)
    if t.is_alive():
        waited = int(time.time() - started)
        recent = " ".join(chatter[-6:]).lower()
        if "error contacting" in recent or "retrying in" in recent:
            return "unreachable", f"could not reach the WebAPI in {waited}s"
        return "timeout", f"still running after {waited}s"
    return result.get("status", "error"), result.get("detail", "")


def stage_retrieve(ECMWFService, args) -> str:
    say(INFO, f"--- stage 2: retrieval  (class={args.klass}, {args.levtype}, "
              f"{args.date} {args.time}Z, one parameter)")
    target = Path(args.out)
    if target.exists():
        target.unlink()

    req = tiny_request(args.klass, args.date, args.time, args.levtype, args.grid)
    say(INFO, "    " + "  ".join(f"{k}={v}" for k, v in req.items()))

    status, detail = retrieve(ECMWFService, req, target, args.timeout)

    if status == "unreachable":
        say(BAD, f"could not reach the ECMWF WebAPI at all ({detail}).")
        say(INFO, "    This is not a queue and not a licence problem - the host")
        say(INFO, "    is unreachable. On eX3 check the `url` in ~/.ecmwfapirc,")
        say(INFO, "    then whether egress to api.ecmwf.int is permitted:")
        say(INFO, "      sbatch bris/slurm/check_egress.sbatch")
        return "unreachable"

    if status == "timeout":
        say(WARN, f"no answer within {args.timeout}s.")
        say(INFO, "    That is not itself a refusal - MARS queues, and tape can")
        say(INFO, "    be slow. Re-run with --timeout 1800, or check the queue")
        say(INFO, "    at https://apps.ecmwf.int/mars-activity/")
        return "timeout"

    if status == "ok" and target.exists() and target.stat().st_size > 0:
        say(OK, f"retrieved {target.stat().st_size:,} bytes -> {target}")
        return "ok"

    if status == "no_service":
        say(BAD, "the MARS service is not enabled for this account.")
        for line in detail.splitlines()[:6]:
            say(INFO, "    " + line)
        say(INFO, "    Authentication succeeded - the server greeted you by name.")
        say(INFO, "    What is missing is authorisation for the service itself,")
        say(INFO, "    which is granted per account and is NOT implied by having")
        say(INFO, "    an ECMWF login. No class will work until it is on.")
        return "no_service"

    if status == "refused":
        say(BAD, "the request was refused.")
        for line in detail.splitlines()[:12]:
            say(INFO, "    " + line)
        return "refused"

    say(BAD, "the request failed.")
    for line in detail.splitlines()[:12]:
        say(INFO, "    " + line)
    return "error"


# --- stage 3 ----------------------------------------------------------------

def grib_points(path: Path) -> tuple[int | None, dict]:
    """Point count and a few keys, by whatever means are available."""
    info: dict = {}
    try:
        import eccodes
        with path.open("rb") as fh:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                return None, info
            for key in ("numberOfDataPoints", "gridType", "shortName", "N",
                        "dataDate", "dataTime", "level", "typeOfLevel"):
                try:
                    info[key] = eccodes.codes_get(gid, key)
                except Exception:                     # noqa: BLE001
                    pass
            eccodes.codes_release(gid)
        return info.get("numberOfDataPoints"), info
    except ImportError:
        pass

    import shutil
    import subprocess
    if shutil.which("grib_ls"):
        try:
            out = subprocess.run(
                ["grib_ls", "-p", "numberOfDataPoints,gridType,shortName", str(path)],
                capture_output=True, text=True, timeout=60)
            info["grib_ls"] = out.stdout.strip()
            for tok in out.stdout.split():
                if tok.isdigit() and int(tok) > 1000:
                    return int(tok), info
        except Exception:                             # noqa: BLE001
            pass
    return None, info


def stage_grid(path: Path) -> bool:
    say(INFO, "--- stage 3: grid")
    head = path.open("rb").read(4)
    if head != b"GRIB":
        say(WARN, f"file does not start with the GRIB magic (got {head!r}).")
        say(INFO, "    Retrieval succeeded but the payload is not what we expect.")
        return False
    say(OK, "payload is GRIB")

    n, info = grib_points(path)
    for k, v in info.items():
        if k != "grib_ls":
            say(INFO, f"    {k} = {v}")

    if n is None:
        say(WARN, "no eccodes and no grib_ls - cannot count points here.")
        say(INFO, "    Run this on eX3 inside the dataset env, where anemoi")
        say(INFO, "    pulls eccodes in. Size alone does not prove the grid.")
        return True

    say(INFO, f"    numberOfDataPoints = {n:,}")
    if n == N320_POINTS:
        say(OK, f"N320 reduced Gaussian. {n:,} - {CUTOUT_DROPS:,} dropped by the "
                f"cutout = {n - CUTOUT_DROPS:,}, which is the checkpoint's count.")
        return True

    say(BAD, f"expected {N320_POINTS:,} for N320, got {n:,}.")
    say(INFO, "    A dataset built on this grid will not match the checkpoint")
    say(INFO, "    graph. Check that `grid` survived into the request and that")
    say(INFO, "    the server did not fall back to a regular lat/lon grid.")
    return False


def report_other_routes() -> None:
    """What else is configured, so the next question is a precise one."""
    say(INFO, "--- what this account has configured")

    cds = Path.home() / ".cdsapirc"
    if cds.exists():
        say(OK, f"{cds} exists - the CDS route, which is how the ERA5 build")
        say(INFO, "       actually worked. `class: ea` via use_cdsapi_dataset does")
        say(INFO, "       NOT go through services/mars, so it is unaffected by this.")
        say(INFO, "       Verify it independently:")
        say(INFO, "         python3 scripts/test_cds_request.py")
    else:
        say(WARN, f"no {cds} - the ERA5 route is not configured on this host either.")

    rc = Path.home() / ".ecmwfapirc"
    if rc.exists():
        say(OK, f"{rc} exists and authenticates, but carries no MARS role.")

    say(INFO, "")
    say(INFO, "    Three separate grants are easy to confuse, and 'MARS access'")
    say(INFO, "    names only the first:")
    say(INFO, "      1. services/mars on api.ecmwf.int  - the archive. NOT on.")
    say(INFO, "      2. anemoi.ecmwf.int/datasets       - the training corpus.")
    say(INFO, "      3. CDS / cds.climate.copernicus.eu - ERA5. Public.")
    say(INFO, "    A grant of 2 does not enable 1. Worth establishing which")
    say(INFO, "    arrived before rebuilding anything on the assumption of od.")


# --- main -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="klass", default="od", choices=["od", "ea"],
                    help="od = IFS operational analysis (default). This is the "
                         "one Bris was trained on and the one worth probing.")
    ap.add_argument("--date", default="2025-04-01", help="YYYY-MM-DD")
    ap.add_argument("--time", default="00", help="analysis hour, e.g. 00 or 12")
    ap.add_argument("--levtype", default="sfc", choices=["sfc", "pl"])
    ap.add_argument("--grid", default="N320")
    ap.add_argument("--timeout", type=int, default=600,
                    help="seconds before giving up (default 600)")
    ap.add_argument("--out", default="/tmp/mars_probe.grib")
    ap.add_argument("--keep", action="store_true", help="do not delete the probe file")
    args = ap.parse_args()

    print(f"MARS access probe - class={args.klass} {args.date} {args.time}Z "
          f"{args.levtype} {args.grid}\n")

    ok, ECMWFService = stage_env()
    if not ok:
        say(INFO, "\nVERDICT: setup is incomplete. Nothing was requested.")
        return 2
    print()

    if not stage_identity(ECMWFService):
        say(INFO, "\nVERDICT: credentials did not resolve. Nothing was requested.")
        return 2
    print()

    status = stage_retrieve(ECMWFService, args)
    print()

    if status == "ok":
        target = Path(args.out)
        grid_ok = stage_grid(target)
        if not args.keep:
            target.unlink(missing_ok=True)
        print()
        if grid_ok:
            say(INFO, f"VERDICT: class {args.klass} is open and the grid is right.")
            say(INFO, "         Generate the recipe with:")
            say(INFO, "           python3 scripts/make_era5_recipe.py \\")
            say(INFO, "             $BRIS_MODEL_DIR/ckpt-metadata.json --source od")
            say(INFO, "         Then probe levtype pl too: --levtype pl")
            return 0
        say(INFO, f"VERDICT: class {args.klass} is open, but the grid is wrong.")
        say(INFO, "         Do not build a dataset until stage 3 passes.")
        return 1

    if status == "unreachable":
        say(INFO, "VERDICT: no route to the WebAPI. Nothing about your licence")
        say(INFO, "         was established - fix connectivity and re-run.")
        return 2

    if status == "timeout":
        say(INFO, "VERDICT: inconclusive - the request is still queued. It was")
        say(INFO, "         accepted, which already rules out an outright")
        say(INFO, "         refusal; re-run with a longer --timeout to see it")
        say(INFO, "         through.")
        return 3

    if status == "no_service":
        say(INFO, "VERDICT: the client, the credentials and the network are all")
        say(INFO, "         fine - you were greeted by name. The MARS service is")
        say(INFO, "         simply not enabled for this account, so no class is")
        say(INFO, "         reachable through it and no retry will change that.")
        print()
        report_other_routes()
        return 1

    # A class-specific refusal, with the service itself reachable. Here a second
    # class IS a meaningful control.
    if args.klass == "od":
        say(INFO, "--- fallback: same request against class ea, to tell a licence")
        say(INFO, "    problem apart from a broken client or network")
        fb = argparse.Namespace(**vars(args))
        fb.klass = "ea"
        fb.out = args.out + ".ea"
        fb_status = stage_retrieve(ECMWFService, fb)
        Path(fb.out).unlink(missing_ok=True)
        print()
        if fb_status in ("unreachable", "timeout"):
            say(INFO, "VERDICT: inconclusive - the fallback did not resolve either.")
            return 3
        if fb_status == "ok":
            say(INFO, "VERDICT: the client and credentials work, but class od is")
            say(INFO, "         not open to you. The grant you received may be")
            say(INFO, "         the anemoi catalogue rather than MARS od - they")
            say(INFO, "         are separate. Ask which one before rebuilding.")
            return 1
        say(INFO, "VERDICT: ea failed too, so this is the client, the credentials")
        say(INFO, "         or the network - not the od licence. Fix that first.")
        return 2

    say(INFO, "VERDICT: the request failed. See the message above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
