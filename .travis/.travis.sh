#!/bin/bash
#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on OS X
    # e.g. brew install pyenv-virtualenv
    brew install pyenv-virtualenv
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    pyenv virtualenv $TOXENV py$TOXENV
    pyenv activate py$TOXENV
fi

pip install -r requirements.txt
