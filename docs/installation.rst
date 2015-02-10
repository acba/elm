============
Installation
============

At the command line::

    $ pip install elm

Or, if you have virtualenv installed::

    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install elm


.. note::
    If you found an error while using :func:`ELMKernel.search_param` or
    :func:`ELMRandom.search_param`, probably is because **pip** installed an
    outdated version of **optunity** (currently their pip package and github are not synced).

To fix it, do::

     # Download package
     $ pip install -d . elm
     # Unzip it
     $ tar -xf elm*.tar.gz
     # Install requirements.txt
     $ cd elm*; pip install -r requirements.txt
