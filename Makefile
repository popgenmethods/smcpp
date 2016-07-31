GIT_VERSION=$(shell git describe --abbrev=4 --always --tags)

all: update_version
	GIT_VERSION="$(GIT_VERSION)" ARCHFLAGS="-arch x86_64" \
				PATH="/usr/local/opt/ccache/libexec:/usr/lib/ccache:${PATH}" \
				python setup.py develop
clean:
	rm -rf *.so *.pyc __pycache__ test/*.pyc test/__pycache__ src/*.o build .moran.dat*
update_version:
	echo 'version="$(GIT_VERSION)"' > smcpp/version.py
