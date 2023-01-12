"""安装文件"""
from setuptools import setup

import npml

metadata = dict(
    name="npml",
    maintainer="chenxilin",
    maintainer_email="635943647@qq.com",
    description="A machine learning package implemented by numpy",
    license="MIT License",
    url="https://github.com/xiligey/npml",
    download_url="https://pypi.org/project/npml/#files",
    version=npml.__version__,
    python_requires=">=3.8",
    install_requires=["numpy", "setuptools", "pandas", "matplotlib"],
    tests_require=["pytest"],
    package_data={"": ["*.csv", "*.gz", "*.txt", "*.pxd", "*.rst", "*.jpg"]},
    zip_safe=False
)

setup(**metadata)
