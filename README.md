# 🛒 DealSense AI

An intelligent agentic AI system that scans, evaluates, and discovers the best online deals in real-time. DealSense AI combines multiple AI models and specialized agents to predict product prices, identify significant discounts, and filter out irrelevant deals — ensuring users only see genuinely valuable offers.

## ✨ Core Features

- **🧠 Central Planning Agent**: Orchestrates the complete workflow from deal scanning to price prediction and filtering
- **🔍 Real-time Deal Discovery**: Fetches live deals from RSS feeds with automatic duplicate detection
- **💰 Advanced Price Prediction**: Multi-model ensemble including OpenAI GPT, fine-tuned LLaMA, XGBoost, and RAG pipeline
- **⚡ Smart Filtering**: Automated deal filtering based on configurable discount thresholds
- **📊 Transparent Operations**: Complete logging of all agent actions and decision-making processes
- **🎯 Structured Output**: Clean summaries with price estimates, discount percentages, and direct links
- **🧩 Memory System**: Shared storage prevents processing duplicate deals
- **☁️ Scalable Execution**: Compute-intensive models run remotely on Modal for optimal performance

## 🚀 Tech Stack

- **Languages & Frameworks**: Python, FastAPI, React
- **AI & Machine Learning**: 
  - OpenAI GPT for deal selection and RAG-based price prediction
  - Fine-tuned LLaMA 3.1 8B (QLoRA) for specialized price estimation
  - XGBoost regression with E5 embeddings
  - RAG pipeline using ChromaDB vector database
  - Linear regression ensemble model

## 🏗️ Architecture

DealSense AI employs a multi-agent architecture with specialized components:

### 🤖 Specialized Agents

- **Deal Scanner Agent**: Fetches and filters deals from RSS feeds using OpenAI GPT
- **Fine-Tuned LLM Agent**: Price prediction using custom-trained LLaMA model on Modal
- **RAG-Based Agent**: Contextual price estimation with retrieval-augmented generation
- **XGBoost Agent**: Regression-based predictions using product embeddings
- **Ensemble Agent**: Combines all predictions for final price estimates

### 📊 Models & Data

- **Dataset**: 409K curated items from Amazon Reviews 2023 across 8 product categories
- **Embeddings**: E5-small-v2 model with ChromaDB for similarity search
- **Training**: Balanced sampling with price-based stratification
- **Storage**: Models and datasets hosted on Hugging Face Hub

## 🛠️ Setup

1. **Environment Variables**
   ```bash
   OPENAI_API_KEY=your_openai_key
   MODAL_TOKEN_ID=your_modal_token_id
   MODAL_TOKEN_SECRET=your_modal_token_secret
   ```

## 📈 How It Works

1. **Scan**: Deal Scanner Agent fetches real-time deals from RSS feeds
2. **Predict**: Multiple AI agents generate independent price predictions
3. **Ensemble**: Linear regression model combines predictions for final estimate
4. **Filter**: Deals below discount threshold are automatically filtered out
5. **Present**: Users receive structured summaries of valuable deals only

## 🎯 Use Cases

- **Smart Shopping**: Automatically discover genuinely discounted products
- **Price Monitoring**: Track market prices across multiple categories
- **Deal Validation**: Verify if advertised discounts represent real value
- **Market Research**: Analyze pricing patterns and trends

## 📖 Documentation

For detailed technical documentation, architecture deep-dives, and implementation guides, visit:

- **[Models & AI Components](docs/models.md)** - Fine-tuned LLaMA, XGBoost, RAG pipeline, and ensemble details
- **[Features & Agents](docs/features.md)** - Complete guide to all specialized agents and their capabilities  
- **[Dataset & Curation](docs/data.md)** - Data sourcing, filtering, sampling, and embedding processes
- **[API Integration](docs/api.md)** - Internal API usage, authentication, and service integrations

---

*DealSense AI leverages cutting-edge AI to transform how people discover and evaluate online deals, ensuring every recommendation represents genuine value.*