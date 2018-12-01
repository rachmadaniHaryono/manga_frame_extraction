import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="manga-frame-extraction",
    version="0.0.1",
    author="DWANGO Co.",
    #  author_email="author@example.com",
    description="Extract manga frame.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DwangoMediaVillage/manga_frame_extraction",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
