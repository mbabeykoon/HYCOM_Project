from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirments(file_path:str)-> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirments = []
    with open('requirements.txt') as file_obj:
        requirments = file_obj.readlines()
        requirments = [req.replace("\n","" ) for req in requirments]

        if HYPEN_E_DOT in requirments:
            requirments.remove(HYPEN_E_DOT)
    return requirments


setup(name='ts_prediction',
      version='0.0.1',
      author= 'Madusanka Madiligama',
      author_email='mbabeykoon@gmail.com',
      packages=find_packages(),
      install_requires = get_requirments('requirments.txt'))