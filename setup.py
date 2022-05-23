import setuptools


def _requires_from_file(filename):
    return open(filename).read().splitlines()


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QuestionRecommender",
    version="0.0.1",
    author="diamond",
    author_email="abc@example.copm",
    description="It's pip... with git.",
    long_description=long_description,
    url="https://github.com/hossy37/DAC21_Diamond",
    install_requires=_requires_from_file("requirements.txt"),
    packages=setuptools.find_packages("./src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
