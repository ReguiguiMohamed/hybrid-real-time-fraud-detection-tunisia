from setuptools import setup, find_packages

setup(
    name="hybrid-real-time-fraud-detection",
    version="1.0.0",
    description="Hybrid Real-Time Fraud Detection System for Tunisia",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyspark==4.1.1",
        "delta-spark==4.0.1",
        "xgboost==3.1.3",
        "confluent-kafka==2.3.0",
        "faker==40.1.2",
        "pydantic==2.12.5",
        "python-dotenv==1.0.0",
        "chromadb>=0.5.0",
        "sentence-transformers>=3.0.1",
        "requests>=2.31.0",
        "google-generativeai==0.8.6",
        "fastapi>=0.115.0",
        "uvicorn>=0.32.0",
        "streamlit>=1.39.0",
        "plotly>=5.24.0",
        "pandas>=2.2.0",
        "scipy>=1.14.0",
        "httpx>=0.27.0",
    ],
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "tunisia-fraud-producer=producer.producer:main",
        ],
    },
)