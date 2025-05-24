setup:
	uv sync

build:
	uv run maturin develop -r --manifest-path src/rtmdet-object-detection/Cargo.toml

test-python:
	uv run -m pytest tests/

test-rust:
	cargo test --manifest-path src/rtmdet-object-detection/Cargo.toml

test: test-python test-rust