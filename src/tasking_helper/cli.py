"""
Command-line interface for tasking_helper.

Run with:
    python -m tasking_helper <subcommand> [options]
    tasking-helper <subcommand> [options]       # after pip install

Subcommands
-----------
epoch       Parse or format Julian Date / epoch strings.
satcat      Generate a LOST-format satellite catalog (wraps make_satcat.py).
covariance  Generate position covariance files (wraps make_covariance.py).
"""

import argparse
import pathlib
import subprocess
import sys

from tasking_helper.utils.jdate import fmt_epoch, jd_to_datetime, parse_epoch

# Paths to the standalone scripts bundled with the package.
_SCRIPTS: dict[str, pathlib.Path] = {
    "satcat": pathlib.Path(__file__).parent / "make_satcat.py",
    "covariance": pathlib.Path(__file__).parent / "make_covariance.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tasking-helper",
        description="Satellite tasking utilities.",
    )
    sub = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    sub.required = True

    # ------------------------------------------------------------------ epoch
    ep = sub.add_parser(
        "epoch",
        help="Parse or format epoch strings / Julian Dates.",
        description="Convert between epoch strings and Julian Dates.",
    )
    ep_action = ep.add_subparsers(dest="epoch_action", metavar="ACTION")
    ep_action.required = True

    parse_p = ep_action.add_parser(
        "parse",
        help="Parse an epoch string and print the Julian Date and UTC time.",
        description=(
            "Parse an ISO 8601 or TLE-format epoch string and print the "
            "equivalent Julian Date and UTC datetime."
        ),
    )
    parse_p.add_argument(
        "epoch",
        metavar="EPOCH",
        help=(
            "Epoch string in ISO 8601 format ('YYYY-mm-ddTHH:MM:SS[.ffffff]', "
            "'YYYY-mm-dd') or TLE format ('yyddd.ddddddd')."
        ),
    )

    fmt_p = ep_action.add_parser(
        "format",
        help="Format a Julian Date as a UTC string.",
        description="Convert a Julian Date to a human-readable UTC string.",
    )
    fmt_p.add_argument(
        "jd",
        type=float,
        metavar="JD",
        help="Julian Date to format.",
    )
    fmt_p.add_argument(
        "--fmt",
        default="%Y-%m-%dT%H:%M:%S",
        metavar="FMT",
        help="strftime format string (default: '%%Y-%%m-%%dT%%H:%%M:%%S').",
    )

    # ----------------------------------------------------------------- satcat
    satcat_p = sub.add_parser(
        "satcat",
        help="Generate a LOST-format satellite catalog from TLE data.",
        description=(
            "Generate a satellite catalog CSV from a TLE file. "
            "All options are forwarded to make_satcat.py; run with --help for details."
        ),
        add_help=False,
    )
    satcat_p.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim to make_satcat.py.",
    )

    # -------------------------------------------------------------- covariance
    cov_p = sub.add_parser(
        "covariance",
        help="Generate position covariance files for a TLE catalog.",
        description=(
            "Generate covariance files for a TLE catalog using the COVGEN model. "
            "All options are forwarded to make_covariance.py; run with --help for details."
        ),
        add_help=False,
    )
    cov_p.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim to make_covariance.py.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ epoch
    if args.subcommand == "epoch":
        if args.epoch_action == "parse":
            try:
                jd = parse_epoch(args.epoch)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            dt = jd_to_datetime(jd)
            print(f"JD  : {jd:.6f}")
            print(f"UTC : {dt.strftime('%Y-%m-%dT%H:%M:%S')}")

        elif args.epoch_action == "format":
            print(fmt_epoch(args.jd, fmt=args.fmt))

    # ---------------------------------------------------- script delegation
    elif args.subcommand in _SCRIPTS:
        script = _SCRIPTS[args.subcommand]
        fwd = args.args
        # Strip a leading '--' separator used for shell passthrough convention.
        if fwd and fwd[0] == "--":
            fwd = fwd[1:]
        result = subprocess.run([sys.executable, str(script)] + fwd)
        return result.returncode

    return 0
