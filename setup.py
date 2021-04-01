"""npml安装"""
from setuptools import setup, find_packages
setup(
    name="npml",
    version="1.0",
    packages=find_packages(),
    scripts=['npml/base.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['docutils>=0.3'],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.csv'],
        # And include any *.msg files found in the 'hello' package, too:
        'hello': ['*.msg'],
    },

    # metadata for upload to PyPI
    author="XiLin Chen",
    author_email="635943647@qq.com",
    description="Fastseer Package",
    license="PSF",
    keywords="fastseer",
    url="http://code.bizseer.com/algorithm/fastseer-core",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
