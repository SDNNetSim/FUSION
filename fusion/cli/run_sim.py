# fusion/cli/run_sim.py

from fusion.cli.main_parser import build_parser
from fusion.sim.batch_runner import run_simulation


def main():
    """
    Entrypoint for running simulations from the command line.
    Parses arguments and delegates to the simulation batch runner.
    """
    parser = build_parser()
    args = parser.parse_args()

    # TODO: Recall that "stop flag" was removed here by GPT.
    run_simulation(args)


if __name__ == "__main__":
    main()
