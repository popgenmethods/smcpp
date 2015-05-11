python:
	PATH="/usr/lib/ccache:${PATH}" python setup.py build_ext -i
python-debug:
	python-dbg setup_dbg.py build_ext -i
clean:
	rm -rf _expm.so _pypsmcpp.so *.pyc __pycache__ test/*.pyc test/__pycache__ src/*.o build
