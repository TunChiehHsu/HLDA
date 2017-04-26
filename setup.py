
# coding: utf-8

# In[ ]:

from setuptools import setup

setup(
   name='hLDA',
   version='1.0',
   description='Hierarchical topic model with nCRP',
   author='Tun-Chieh Hsu, Xialingzi Jin, Yen-Hua Chen',
   author_email='yen.hua.chen@duke.edu',
   packages=['hLDA'],  #same as name
   install_requires=['numpy', 'scipy.special', 'random', 'collections'], #external packages as dependencies
)

