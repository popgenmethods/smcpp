#!/bin/bash
#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on OS X
    # e.g. brew install pyenv-virtualenv
    brew update
    brew install mpfr gmp gsl pyenv-virtualenv 
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    pyenv virtualenv $TOXENV py$TOXENV
    pyenv activate py$TOXENV
else
    sudo apt-get -qq update
    sudo apt-get install -y libmpc-dev libmpfr-dev libgmp-dev libgsl0-dev 
fi

pip install -r requirements.txt
python setup.py develop
