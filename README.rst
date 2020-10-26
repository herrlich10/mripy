Introduction
============

This is a collection of handy small tools for analyzing neuroimaging data (esp. 
high resolution fMRI), which can be used both as a Python package and 
a set of command line tools. It is a useful augmentation to the ``AFNI`` tool chain.

Installation (in linux/mac)
===========================

Install ``AFNI`` and ``FreeSurfer``
-----------------------------------
Please follow the instructions in their official websites (AFNI_, FreeSurfer_).

.. _AFNI: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/install_instructs/index.html

.. _FreeSurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall

Install ``mripy``
-----------------
Download and install the ``mripy`` package:

.. code-block:: shell

    $ pip install mripy

Set ``$PATH`` for using the package scripts in the terminal. For bash, the commands look like:

.. code-block:: shell

    $ vi ~/.bashrc
    $ export PATH="path/to/mripy/scripts":$PATH

The path could be something like "~/anaconda3/lib/Python3.8/site-packages/mripy/scripts".

Download and install the dependencies:

.. code-block:: shell

    $ pip install nibabel deepdish
    

Install ``neuropythy docker`` (for HCP retinotopy altas)
--------------------------------------------------------
First install `docker <https://www.docker.com/products/docker-desktop>`_ for your OS, then pull the ``neuropythy`` image:

.. code-block:: shell

    # Pull a particular version
    $ docker pull nben/neuropythy@sha256:2541ee29a8d6bc676d9c3622ef4a38a258dd90e06c02534996a1c8354f9ac888

    # Give it a tag
    $ docker tag b38ebfcf6477 nben/neuropythy:mripy

More information about the `HCP retinotopy altas <https://nben.net/HCP-Retinotopy/>`_ and 
the `neuropythy <https://github.com/noahbenson/neuropythy>`_ package can be found in Noah C. Benson's website.


