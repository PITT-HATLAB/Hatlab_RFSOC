from distutils.core import setup

setup(
      name='Hatlab_RFSOC',
      version='0.0.1',
      packages=['Hatlab_RFSOC'],

      python_requires='>=3.7, <4',

      install_requires=[
            "Pyro4", 
            "qick"
      ]
)