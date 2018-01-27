.DEFAULT_GOAL := help
.PHONY: clean clean-test clean-pyc clean-build clean-latex docs help build push all run run-qa


define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

ifeq ("$(shell git status -s | grep -vP '^\?\?')","")
    GIT_VERSION="Git: $(shell git log -n1 --format="%h") on $(shell git rev-parse --abbrev-ref HEAD)\nBuild: $(shell date +%Y-%m-%dT%H:%M:%S)"
else
    $(info !                                                                    !)
    $(info !            !!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!             !)
    $(info !            !!                                       !!             !)
    $(info !            !!          BUILDING DIRTY VERSION       !!             !)
    $(info !            !!                                       !!             !)
    $(info !            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!             !)
    $(info !                                                                    !)
    GIT_VERSION="!! DIRTY !!\nGit: $(shell git log -n1 --format="%h") on $(shell git rev-parse --abbrev-ref HEAD)\nBuild: $(shell date +%Y-%m-%dT%H:%M:%S)"
endif


help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-latex

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-latex: ## remove latex artifacts
	find . -name '*.aux' -exec rm -fr {} +
	find . -name '*.dvi' -exec rm -fr {} +
	find . -name '*.fdb_latexmk' -exec rm -fr {} +
	find . -name '*.fls' -exec rm -fr {} +
	find . -name '*.log' -exec rm -fr {} +
	find . -name '*.toc' -exec rm -fr {} +
	find . -name '*.xdv' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/

lint: ## check style with flake8
	flake8 src tests

test:
	py.test tests/

run-local:
	python ./manage.py runserver --pythonpath etc/hack6 --settings settings_local

coverage: ## check code coverage quickly with the default Python
	coverage run --source src setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

doc/report/report.pdf: $(wildcard doc/report/*) $(wildcard doc/report/**/*)
	cd doc/report; latexmk -xelatex report.tex

report: doc/report/report.pdf

doc/final/final.pdf: $(wildcard doc/final/*) $(wildcard doc/final/**/*)
	cd doc/final; latexmk -xelatex final.tex

final: doc/final/final.pdf

release: clean ## package and upload a release
	python setup.py sdist upload
	python setup.py bdist_wheel upload

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

run-db:
	docker-compose -f docker-compose/local/docker-compose.yml up

restore-db: ## TODO
	pg_restore --verbose --clean --no-acl --no-owner -h localhost -U postgres -d postgres data/heroku_dump/latest.dump

psql:
	psql -h localhost -U postgres postgres

