from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    Returns list of requirements from a file, ignoring '-e .'
    '''
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and req.strip() != HYPHEN_E_DOT]
    return requirements

setup(
    name='mlproject',
    version='0.1.0',
    author='Ajith Reddy',
    author_email='ajithreddy.pr@gmail.com',
    description='Churn Prediciton Model',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    include_package_data=True
)
