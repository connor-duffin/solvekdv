import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="solvekdv",
    version="0.0.1",
    author="Connor Duffin",
    author_email="connor.p.duffin@gmail.com",
    description="Solve the KdV equation using finite differences.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
