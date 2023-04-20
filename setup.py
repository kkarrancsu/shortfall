import setuptools
import os

# with open("README.md", "r") as fh:
#     long_description = fh.read()

__author__ = 'Alex North, Kiran Karra, Tom Mellan'
__email__ = 'anorth@protocol.ai, kiran.karra@protocol.ai, tom.mellan@protocol.ai'
__version__ = '0.1.0'

install_requires = ['pandas',
                    'jax',
                    'tqdm',
                    ]

setuptools.setup(
    name="shortfall",
    version=__version__,

    description='Model of Shortfall proposal',
    long_description="Model of Shortfall proposal",
    long_description_content_type="text/markdown",

    url = 'https://github.com/kkarrancsu/shortfall',

    author=__author__,
    author_email=__email__,

    license='MIT License',

    python_requires='>=3',
    #packages=['shortfall'],
    packages=setuptools.find_packages(),

    install_requires=install_requires,

    zip_safe=False
)