all:
	ARCHFLAGS="-arch x86_64" PATH="/usr/local/opt/ccache/libexec:/usr/lib/ccache:${PATH}" python setup.py develop
clean:
	rm -rf *.so *.pyc __pycache__ test/*.pyc test/__pycache__ src/*.o build .moran.dat*
pyi: update_version
	rm -rf build dist
	pyinstaller --log-level DEBUG -p $(VIRTUAL_ENV)/local/lib/python2.7 --onedir \
		--clean -y -n smc++-$(GIT_VERSION)-$(HWNAME) \
		--debug \
		scripts/smc++
wheel:
	python setup.py bdist_wheel
