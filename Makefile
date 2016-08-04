GIT_VERSION=$(shell git describe --abbrev=4 --always --tags | sed 's/-/+/')
HWNAME=$(shell uname -ms | tr '[:upper:]' '[:lower:]' | tr ' ' -)

all: update_version
	GIT_VERSION="$(GIT_VERSION)" ARCHFLAGS="-arch x86_64" \
				PATH="/usr/local/opt/ccache/libexec:/usr/lib/ccache:${PATH}" \
				python setup.py develop
clean:
	rm -rf *.so *.pyc __pycache__ test/*.pyc test/__pycache__ src/*.o build .moran.dat*
pyi: update_version
	rm -rf build dist
	pyinstaller --log-level DEBUG -p $(VIRTUAL_ENV)/local/lib/python2.7 --onedir \
		--clean -y -n smc++-$(GIT_VERSION)-$(HWNAME) \
		--debug \
		scripts/smc++
update_version:
	echo "__version__ = '$(GIT_VERSION)'" > smcpp/_version.py

wheel:
	GIT_VERSION="$(GIT_VERSION)" python setup.py bdist_wheel
