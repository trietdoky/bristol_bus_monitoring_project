from setuptools import setup, find_packages

setup(
    name='bods_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas'
    ],
    entry_points={
        'console_scripts': [
            'bods_project=bods_project.module1:main',
        ],
    },
    author='Triet Do',
    author_email='trietdoky@gmail.com',
    description='BODS tracking project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/trietdoky/bods_project',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)