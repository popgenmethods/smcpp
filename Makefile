all:
	PATH=/usr/local/opt/ccache/libexec:/usr/lib/ccache:${PATH} python setup.py develop
clean:
	python setup.py clean --all
current_tag:
	git tag -f -m 'current tag' -a current
	git push -f --tags pgm
wheel:
	python setup.py bdist_wheel
