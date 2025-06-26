from setuptools import setup, find_packages
import os

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith('#')]

setup(
    name="credit_health_engine",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated backend system for credit agent classification and monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/credit-health-engine",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords='credit risk analysis, agent classification, financial monitoring',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/credit-health-engine/issues',
        'Source': 'https://github.com/yourusername/credit-health-engine',
    },
)
