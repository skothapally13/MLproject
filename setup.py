from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:   
    """
    This function returns the list of requirements from requirements.txt
    """

    requirements = []

    with open(file_path, "r") as file:
        requirements = [
            requirement.strip()
            for requirement in file.readlines()
            if requirement.strip()
        ]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Sandhya",
    author_email="*****@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)