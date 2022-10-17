lint:
	poetry run python3 -m mypy --install-types
	poetry run python3 -m mypy src/ main.py

fmt:
	poetry run python3 -m black src/ tests/ scripts/ main.py
	poetry run python3 -m isort ./src ./tests main.py salvage_video.py calculate_error_statistics.py

run:
	poetry run python3 main.py --config_file ./configs/gallego_cvpr_2018_contrast_maximization.yaml

test:
	poetry run pytest -vvv -c tests/ -o "testpaths=tests" -W ignore::DeprecationWarning

integration-test:
	poetry run python3 main.py --config_file ./configs/integration_tests/1.yaml
	poetry run python3 main.py --config_file ./configs/integration_tests/2.yaml
