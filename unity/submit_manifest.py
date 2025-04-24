"""
Read manifest.csv, patch template, and submit a SLURM array.
"""
import argparse
import pathlib
import os
import subprocess
import tempfile


def parse_cli():
    """
    Parse CLI arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--exp', required=True, help='e.g. 20250424_nsfnet')
    p.add_argument('--network', required=True)
    return p.parse_args()


def main():
    """
    Controls the script.
    """
    args = parse_cli()
    exp_dir = pathlib.Path(args.exp)
    manifest = exp_dir / 'manifest.csv'
    rows = sum(1 for _ in open(manifest, encoding='utf-8')) - 1  # pylint: disable=consider-using-with
    job_dir = exp_dir / 'jobs'
    job_dir.mkdir(exist_ok=True)

    # TODO: This is not where this is located, why is it called sbatch and not .sh?
    tpl_path = pathlib.Path('bash_scripts', 'run_rl_sim.sh')
    tpl_txt = tpl_path.read_text(encoding='utf-8')
    sbatch_txt = (tpl_txt
                  .replace('__N_JOBS__', str(rows - 1))
                  .replace('__JOB_DIR__', str(job_dir)))

    sbatch_file = tempfile.NamedTemporaryFile('w', delete=False,
                                              suffix='.sh')  # pylint: disable=consider-using-with
    sbatch_file.write(sbatch_txt)
    sbatch_file.close()

    env = os.environ.copy()
    env.update({
        'MANIFEST': str(manifest),
        'N_JOBS': str(rows - 1),
        'JOB_DIR': str(job_dir),
        'NETWORK': args.network,
        'DATE': args.exp.split('_')[0],
    })

    print('Submitting array…')
    print(sbatch_file.name)
    subprocess.run(['sbatch', sbatch_file.name], check=True, env=env)
    print(f'Logs → {job_dir}')


if __name__ == '__main__':
    main()
