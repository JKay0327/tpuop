from setuptools import setup, find_packages

setup(
    name='tpuop',
    version='1.0',
    packages=find_packages(),
    package_data={'tpuop': ['lib/*.so']},
    include_package_data=True,
    install_requires=[
        # 添加任何其他依赖项
    ],
    dependency_links=[
        # 如果需要指定依赖的链接，可以在这里添加
    ],
)