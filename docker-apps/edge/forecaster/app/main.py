import argparse
import logging
import sys

# suppress ssl warning for IoT platform
import urllib3

import setup_edge
import setup_offline

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

EDGE_SUBCOMMAND = "edge"
OFFLINE_SUBCOMMAND = "offline"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title="Subcommands")

    edge_parser = subparser.add_parser(EDGE_SUBCOMMAND)
    edge_parser.set_defaults(subcommand_func=setup_edge.add_arguments)
    edge_parser.set_defaults(subcommand_kwargs={"parser": edge_parser})

    offline_parser = subparser.add_parser(OFFLINE_SUBCOMMAND)
    offline_parser.set_defaults(subcommand_func=setup_offline.add_arguments)
    offline_parser.set_defaults(subcommand_kwargs={"parser": offline_parser})

    sub_args = parser.parse_args(sys.argv[1:2])
    sub_args.subcommand_func(**sub_args.subcommand_kwargs)
    args = parser.parse_args()
    args.func(args)
