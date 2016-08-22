all:
	ARCHFLAGS="-arch x86_64" PATH="/usr/local/opt/ccache/libexec:/usr/lib/ccache:${PATH}" python setup.py develop
clean:
	python setup.py clean --all
wheel:
	python setup.py bdist_wheel
