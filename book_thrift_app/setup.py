from setuptools import setup, find_packages

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='book_thrift_package',
      version="0.1",
      description="Book recommendation system",
      packages=find_packages(),
      install_requires=requirements)
