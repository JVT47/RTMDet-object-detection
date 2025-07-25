from collections.abc import Generator, Iterable
from itertools import islice
from pathlib import Path


def batch_image_files(image_files: Iterable[Path], batch_size: int) -> Generator[list[Path], None, None]:
    """Yield a batch of image file paths."""
    iterator = iter(image_files)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
