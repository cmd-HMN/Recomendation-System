import threading
from argparse import ArgumentParser

from rich.console import Console

from config import cfg
from pipeline import Pipeline

# No predictions needed, only training
args = ArgumentParser(description="This CLI that helps to simplify the training.")
args.add_argument("--model", type=str, required=True, help="Name of the model to train")
args.add_argument("--params", type=dict, default={}, help="Parameters for the model")
args.add_argument("--watchit", default=False, type=bool, help="Enable watchdog")
args.add_argument("--watch_path", default=cfg.USER_DATA_PATH, type=str, help="Path to watch for changes")

args = args.parse_args()
console = Console()

console.rule("[bold blue]Initializing Pipeline[/bold blue]")
pipeline = Pipeline(name=args.model, params=args.params)
console.print(f"[bold green]✔ Pipeline initialized with model '{args.model}'[/bold green]")

if args.watchit:
    from watch_it import Watcher

    watcher = Watcher(Pipeline=pipeline)
    console.print("[bold green]Watchdog is running...[/bold green]")
    observer = watcher.start_watching(path_to_watch=args.watch_path)
    console.print("[bold green]Exiting...[/bold green]")
    exit(0)


t = threading.Thread(target=pipeline.fit, daemon=True)
t.start()

console.rule("[bold blue]Training started[/bold blue]")
with console.status("Waiting...") as status_spinner:
    while t.is_alive():
        status_spinner.update(f"[bold cyan]{pipeline.status}[/bold cyan]")

console.print(f"[bold green]✔ Pipeline trained with status {pipeline.status}![/]")
