all:
	ARCHFLAGS="-arch x86_64" PATH="/usr/local/opt/ccache/libexec:/usr/lib/ccache:${PATH}" python setup.py develop
clean:
	rm -rf *.so *.pyc __pycache__ test/*.pyc test/__pycache__ src/*.o build .moran.dat*
