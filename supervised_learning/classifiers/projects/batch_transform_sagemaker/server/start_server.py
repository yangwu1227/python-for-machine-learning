import subprocess
from multiprocessing import cpu_count


def calculate_gunicorn_workers() -> int:
    """
    Calculate the recommended number of Gunicorn workers based on the number of CPU cores.
    See https://docs.gunicorn.org/en/latest/design.html#how-many-workers for more information.

    Returns
    -------
    int
        The number of Gunicorn workers recommended for optimal performance.
    """
    num_cores = cpu_count()
    # Ensure that we have at least one worker
    num_workers = max(2 * num_cores + 1, 1)
    return num_workers


def main() -> int:
    num_workers = calculate_gunicorn_workers()
    cmd = [
        "gunicorn",
        "-w",
        str(num_workers),
        "-k",
        "uvicorn.workers.UvicornWorker",
        "server.main:app",
        "--bind",
        "0.0.0.0:8080",
        "--timeout",
        "60",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as error:
        print(f"Gunicorn failed to start: {error}")
        return 1

    return 0


if __name__ == "__main__":
    main()
