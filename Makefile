# Define the default target
default: run

# Define a target to run the Python script
run:
	python -m venv venv &&\
	pip install -r requirements.txt &&\
	source venv/bin/activate &&\
	python validate_project2.py &&\
	python src/main.py
