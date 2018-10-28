from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      author='Kubilay Yavuz',
      author_email='fatihkub007@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py',
          'keras',
          'pandas', 'numpy', 'sklearn', 'tensorflow', 'google'

      ],
      zip_safe=False)
