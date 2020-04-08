from pathlib import Path
import os
from setuptools import setup
from setuptools.command.install import install
import shutil
import sys
import urllib.request
import zipfile


class InstallLemmataModels(install):
    """Helper to install CLTK lemmatization models."""
    description = "Install CLTK lemmatization models to $HOME/cltk_tools"

    def run(self):
        """Install CLTK lemmatization models.

        Tesserae uses the CLTK lemmatizers for each language, and these have
        accompanying data that must be installed separately. This function
        installs them to `$HOME/cltk_data` where they may be found by the
        lemmatizer.
        """
        latin = 'https://github.com/cltk/lat_models_cltk/archive/master.zip'
        greek = 'https://github.com/cltk/grc_models_cltk/archive/master.zip'
        home = str(Path.home())


        try:
            # Set up the file paths and directories for the Latin models
            base = os.path.join(home, 'cltk_data', 'lat', 'model')
            if not os.path.isdir(base):
                os.makedirs(base, exist_ok=True)

            # Download the Latin models and move the ZIP archive
            fname = os.path.join(base, 'lat_models_cltk.zip')
            with urllib.request.urlopen(latin) as response, open(fname, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

            # Extract all files from the ZIP archive
            with zipfile.ZipFile(fname, mode='r') as zf:
                zf.extractall(base)

            # Rename the extracted directory to match the expected name
            fname, _ = os.path.splitext(fname)
            os.rename(fname + '-master', fname)
        except OSError:
            pass

        try:
            # Repeat the process with Greek models
            base = os.path.join(home, 'cltk_data', 'grc', 'model')
            if not os.path.isdir(base):
                os.makedirs(base, exist_ok=True)

            fname = os.path.join(base, 'grc_models_cltk.zip')
            with urllib.request.urlopen(greek) as response, open(fname, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

            with zipfile.ZipFile(fname, mode='r') as zf:
                zf.extractall(base)

            fname, _ = os.path.splitext(fname)
            os.rename(fname + '-master', fname)
        except OSError:
            pass

        # Run the standard installer
        install.run(self)


setup(
    name='tesserae',
    version='0.1a1',
    description='Fast multi-text n-gram matching for intertext studies.',
    url='https://github.com/tesserae/tesserae-v5',
    author='Jeff Kinnison',
    author_email='jkinniso@nd.edu',
    packages=['tesserae',
              'tesserae.cli',
              'tesserae.db',
              'tesserae.db.entities',
              'tesserae.matchers',
              'tesserae.tokenizers',
              'tesserae.utils'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Users',
        'Topic :: Digital Humanities :: Classics',
        'Topic :: Digital Humanities :: Text Processing',
        'Topic :: Digital Humanities :: Intertext Matching',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    keywords='machine_learning hyperparameters distributed_computing',
    install_requires=[
        'cltk>=0.1.83',
        'nltk>=3.2.5',
        'numpy>=1.14.0',
        'pymongo>=3.6.1',
        'scipy',
        'tqdm',
    ],
    cmdclass={'install': InstallLemmataModels}
)
