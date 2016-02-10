all:
	ARCHFLAGS="-arch x86_64" PATH="/usr/local/opt/ccache/libexec:/usr/lib/ccache:${PATH}" python setup.py develop --user
clean:
	rm -rf _expm.so _pypsmcpp.so *.pyc __pycache__ test/*.pyc test/__pycache__ src/*.o build .moran.dat*
