from setuptools import setup

setup(
    name='tesserae',
    version='0.1a1',
    description='Fast multi-text n-gram matching for intertext studies.',
    url='https://github.com/tesserae/tesserae-v5',
    author='Jeff Kinnison',
    author_email='jkinniso@nd.edu',
    packages=['tesserae',
              'tesserae.db',
              'tesserae.db.entities',
              'tesserae.text_access',
              'tesserae.tokenizers',
              'tesserae.tokenizers.languages',
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
    ]
)
