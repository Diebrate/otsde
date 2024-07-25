from setuptools import setup, find_packages

setup(
    name='otsde',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
    description='Generative modeling using optimal transport and stochastic differential equations',
    author='Kevin Zhang',
    author_email='kevinzhang961030@gmail.com',
    license='MIT'
)