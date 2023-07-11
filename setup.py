from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='saude-menta-tech',
    version='0.1',
    author='Jesse Rodrigues',
    author_email='jessee.orodrigues@gmail.com',
    description='ML usada na Saúde Mental da Indústria Tech',
    packages=find_packages(),
    install_requires=requirements,
)
