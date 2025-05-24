from itertools import islice
from pathlib import Path
from typing import Generator, Iterable


def batch_image_files(
    image_files: Iterable[Path], batch_size: int
) -> Generator[list[Path], None, None]:
    """
    Generator that yields a batch of image file paths.
    """
    iterator = iter(image_files)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
