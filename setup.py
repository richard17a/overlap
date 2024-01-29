"""
setup module docstring
"""

from setuptools import setup
setup(
    name='overlap',
    version='0.0.1',
    author='Richard Anslow',
    author_email='r.anslow@outlook.com',
    url='httpe://github.com/richard17a/overlap',
    license='MIT License',
    packages=['overlap',
              'overlap.atmos',
              'overlap.montecarlo',
              'overlap.craters'],
    description='description tbd.',
    install_requires=[
        # 'python>=3.6.0',
        # 'pandas>=0.10.0'
    ]
)
