from setuptools import setup, find_packages

setup(
    name="llm-token-optimizer",
    version="1.0.0",
    description="Token cost control and auto-optimization for LLM apps — compress prompts, estimate costs, enforce budgets, route to cheap models, and cut LLM spend by up to 60%",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maheshmakvana/llm-token-optimizer",
    packages=find_packages(exclude=["tests*", "venv*"]),
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0",
    ],
    extras_require={
        "tiktoken": ["tiktoken>=0.5"],
        "dev": ["pytest>=7.0", "pytest-asyncio>=0.21"],
        "all": ["tiktoken>=0.5"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "token optimization", "llm cost", "prompt compression",
        "token budget", "llm pricing", "cost control", "openai cost",
        "anthropic cost", "token counting", "llm optimization",
        "prompt optimization", "token reduction", "llm budget",
        "ai cost reduction", "batch api", "model routing",
        "llm efficiency", "token cost", "ai cost optimization",
        "prompt token", "llm token", "context window optimization",
    ],
)
