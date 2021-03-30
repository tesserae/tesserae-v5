import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def get_data():
    """Install CLTK lemmatization models.

        Tesserae uses the CLTK lemmatizers for each language, and these have
        accompanying data that must be installed separately. This function
        installs them to `$HOME/cltk_data` where they may be found by the
        lemmatizer.
        """
    langs = ['lat', 'grc']
    urls = [
        'https://github.com/cltk/lat_models_cltk/archive/master.zip',
        'https://github.com/cltk/grc_models_cltk/archive/master.zip',
    ]
    home = str(Path.home())
    for lang, url in zip(langs, urls):
        try:
            # Set up the file paths and directories for the language models
            base = os.path.join(home, 'cltk_data', lang, 'model')
            if not os.path.isdir(base):
                os.makedirs(base, exist_ok=True)

            final_name = os.path.join(base, lang + '_models_cltk')
            if os.path.exists(final_name):
                # don't download data we already have
                continue

            # Download the Latin models and move the ZIP archive
            fname = os.path.join(base, lang + '_models_cltk.zip')
            with urllib.request.urlopen(url) as response, \
                    open(fname, 'wb') as out_file:
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
            # Install English models
            english = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip'
            base = os.path.join(home, 'nltk_data', 'corpora')
            if not os.path.isdir(base):
                os.makedirs(base, exist_ok=True)

            fname = os.path.join(base, 'wordnet.zip')
            with urllib.request.urlopen(english) as response, open(
                    fname, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

            with zipfile.ZipFile(fname, mode='r') as zf:
                zf.extractall(base)

        except OSError:
            pass


class InstallLemmataModels(install):
    """Helper to install CLTK lemmatization models."""
    description = "Install CLTK lemmatization models to $HOME/cltk_data"

    def run(self):
        get_data()
        # Run the standard installer
        install.run(self)


class DevelopLemmataModels(develop):
    """Helper to install CLTK lemmatization models."""
    description = "Install CLTK lemmatization models to $HOME/cltk_data"

    def run(self):
        get_data()
        # Run the developer installer
        develop.run(self)


setup(name='tesserae',
      version='0.1a1',
      description='Fast multi-text n-gram matching for intertext studies.',
      url='https://github.com/tesserae/tesserae-v5',
      author='Jeff Kinnison',
      author_email='jkinniso@nd.edu',
      packages=find_packages(),
      include_package_data=True,
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
          'cltk>=0.1.83', 'nltk>=3.2.5', 'numpy>=1.14.0', 'pymongo>=3.6.1',
          'scipy', 'tqdm', 'natsort', 'six'
      ],
      cmdclass={
          'install': InstallLemmataModels,
          'develop': DevelopLemmataModels,
      })
