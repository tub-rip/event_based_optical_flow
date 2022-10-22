lint:
	poetry run python3 -m mypy --install-types
	poetry run python3 -m mypy src/ main.py

fmt:
	poetry run python3 -m black src/ tests/ main.py
	poetry run python3 -m isort ./src ./tests main.py 

run:
	poetry run python3 main.py --config_file ./configs/mvsec_indoor_no_timeaware.yaml

test:
	poetry run pytest -vvv -c tests/ -o "testpaths=tests" -W ignore::DeprecationWarning
