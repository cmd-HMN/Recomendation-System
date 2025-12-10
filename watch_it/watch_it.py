import threading
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pipeline import Pipeline


class Watcher(FileSystemEventHandler):
    def __init__(self, Pipeline: Pipeline):
        self.Pipeline = Pipeline
        self.last_trained = 0

    def on_modified(self, event):
        if event.is_directory:
            return

        print(f"File {event.src_path} has been modified. Retraining the model...")

        current_time = time.time()
        if current_time - getattr(self, "last_trained", 0) < 10:
            print("Ignoring rapid successive modifications.")
            return

        self.last_trained = current_time
        self.Pipeline.status = "Retraining due to file change"
        try:
            self.Pipeline.fit()
            self.Pipeline.save_model()
        except Exception as e:
            print(f"Error during retraining: {e}")
        self.Pipeline.status = "Idle"
        print("Retraining complete.")

    def start_watching(self, path_to_watch):
        observer = Observer()
        observer.schedule(self, path=path_to_watch, recursive=False)
        observer_thread = threading.Thread(target=observer.start)
        observer_thread.daemon = True
        observer_thread.start()
        print(f"Started watching {path_to_watch} for changes.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
        print("Stopped watching for file changes.")
        return self
