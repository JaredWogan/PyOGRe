from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readmetext = f.read()

if __name__ == "__main__":
    __version__ = "1.0.0"
    exec(open("src/PyOGRe/Version.py").read())
    setup(
        name="PyOGRe",
        long_description=readmetext,
        long_description_content_type="text/markdown",
        version=__version__,
        packages=find_packages(r"src"),
        package_dir={"": r"src"},
    )
