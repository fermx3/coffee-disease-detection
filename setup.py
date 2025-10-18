from setuptools import find_packages
from setuptools import setup

with open("requirements.txt", encoding="utf-8") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="coffeedd",
    version="0.0.1",
    description="Coffee Disease Detection Package",
    license="MIT",
    author="Le Wagon Coffee Team",
    author_email="coffeesolutionapp@gmail.com",
    url="https://github.com/fermx3/coffee-disease-detection",
    install_requires=requirements,
    packages=find_packages(),
    test_suite="tests",
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False,
)
