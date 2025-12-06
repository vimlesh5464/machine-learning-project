from setuptools import setup, find_packages

# Read the README file for long description (optional)
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="FakeNewsClassifier",
    version="0.0.1",
    author="Vimlesh Gupta",
    author_email="vimleshgupta@example.com",  # you can update this
    description="A machine learning package for detecting fake news using NLP and ML pipelines.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vimleshgupta/FakeNewsClassifier",  # optional GitHub link
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "flask",
        "joblib",
        "matplotlib",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
