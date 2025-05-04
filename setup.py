import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tuneparam", ## 소문자 영단어
    version="0.0.1", ##
    author="Heymin Park", ##
    author_email="phm0707@hanyang.ac.kr", ##
    description="PUT THE PACKAGE DESCRIPTION", ##
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smj0324/2025AIC6008-Hyperparam", ##
    packages=setuptools.find_packages(),
    install_requires=[ ## TODO: 추후 의존성 수정
    ],
    extras_require={}, # 선택적 의존성 추가
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)