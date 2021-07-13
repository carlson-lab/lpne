import setuptools

import lpne

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="LPNE Feature Pipeline",
	version=lpne.__version__,
	description="Local field potential feature extraction pipeline",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)
