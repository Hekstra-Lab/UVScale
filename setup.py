from setuptools import setup, find_packages

setup(
    name='uvscaler',
    version='0.0.1',
    author='Kevin M. Dalton',
    author_email='kmdalton@fas.harvard.edu',
    packages=find_packages(),
    description='Universal variational inference algorithms for crystallographic scales',
    install_requires=[
        "reciprocalspaceship>=0.9.1",
        "pyro-ppl",
    ],
    scripts = [
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
)
