from setuptools import setup, find_packages

with open('README.md', 'r') as f:
  description = f.read()

setup(
  name='psordinal',
  version='0.1',
  author='Paul77ms',
  description="OrdinalClassifier package",
  # package_data=find_packages(),
  entry_points={
      'console_scripts': [
      'psordinal = psordinal.main:main',
    ],
  },
  install_requires=[
  ],

  long_description=description,
  long_decription_content_type='text.markdown',
)