import re
from cmaketools import setup


def get_version():
    try:
        with open("CMakeLists.txt") as cmake_file:
            lines = cmake_file.read()
            version = re.search('VERSION\s+([0-9].[0-9].[0-9]+)', lines).group(1)
            return version

    except Exception:
        raise RuntimeError("Package version retrieval failed. "
                           "Most probably something is wrong with this code and "
                           "you should create an issue at https://github.com/mwydmuch/napkinXC")

setup(
    name="napkinxc_tests",
    version=get_version(),
    author="Marek Wydmuch",
    author_email="mwydmuch@cs.put.poznan.pl",
    description="", # TODO
    long_description="", # TODO
    url="https://github.com/mwydmuch/napkinXC",
    keywords=['machine learning', 'extreme', 'multi-class', 'multi-label', 'classification'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    license="MIT License",
    src_dir=".",
    packages=['napkinxc_tests'],
    package_dir={'napkinxc_tests': 'napkinxc_tests'},
    package_data={'napkinxc_tests': ['*']},
    ext_module_dirs=['python'],
    ext_module_hint=r"pybind11_add_module"
)
