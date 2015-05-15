all:
	PATH="/usr/lib/ccache:${PATH}" python setup.py build_ext -i
debug:
	PATH="/usr/lib/ccache:${PATH}" python-dbg setup.py build_ext -i
clean:
	rm -rf _expm.so _pypsmcpp.so *.pyc __pycache__ test/*.pyc test/__pycache__ src/*.o build .moran.dat*
