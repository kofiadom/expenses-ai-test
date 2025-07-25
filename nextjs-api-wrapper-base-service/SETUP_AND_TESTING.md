# NestJS Expense Processing Service - Setup and Testing Guide

## 🚀 Initial Setup

### 1. Install Dependencies
```bash
cd nextjs-api-wrapper-base-service
npm install
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
# LLM Provider Configuration
LLM_PROVIDER=openai  # or 'anthropic'
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Redis Configuration (for BullMQ)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Application Configuration
PORT=3000
UPLOAD_PATH=./uploads
MAX_RETRY_ATTEMPTS=3

# Optional: AWS Bedrock (if using)
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
```

### 3. Start Redis (Required for BullMQ)
```bash
# Using Docker
docker run -d -p 6379:6379 redis:alpine

# Or install Redis locally and start
redis-server
```

### 4. Start the Application
```bash
npm run start:dev
```

## 🧪 Testing the API

### 1. Access Swagger Documentation
Open your browser and go to:
```
http://localhost:3000/api
```

### 2. Test File Upload Endpoint
**POST** `/documents/process`

**Using curl:**
```bash
curl -X POST \
  http://localhost:3000/documents/process \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/receipt.pdf' \
  -F 'userId=test-user-123' \
  -F 'country=Germany' \
  -F 'icp=Global People'
```

**Using Postman:**
1. Set method to POST
2. URL: `http://localhost:3000/documents/process`
3. Body → form-data:
   - `file`: Select your receipt file (PDF/JPG/PNG/TIFF)
   - `userId`: `test-user-123`
   - `country`: `Germany`
   - `icp`: `Global People`

### 3. Check Processing Status
**GET** `/documents/status/{jobId}`

```bash
curl http://localhost:3000/documents/status/test-user-123
```

### 4. Get Processing Results
**GET** `/documents/results/{jobId}`

```bash
curl http://localhost:3000/documents/results/test-user-123
```

## 🔍 Testing Individual Agents

### Test File Classification Only
```bash
curl -X POST \
  http://localhost:3000/documents/classify \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@receipt.pdf' \
  -F 'country=Germany'
```

### Health Check
```bash
curl http://localhost:3000/health
```

## 📋 Current Limitations & TODOs

### ⚠️ Known Issues:
1. **Prompts Changed**: The TypeScript agents have simplified prompts compared to Python version
2. **Missing Implementation**: 
   - `readDocumentContent()` method needs PDF/image to markdown conversion
   - `loadComplianceData()` needs actual compliance database loading
   - `loadExpenseSchema()` needs schema file loading

### 🔧 Required Fixes:

1. **Restore Original Prompts**: Update TypeScript agents to match Python prompts exactly
2. **Implement Document Reading**: Add PDF/image processing to convert to markdown
3. **Add Compliance Data**: Load actual compliance requirements from files/database
4. **Add Expense Schema**: Load expense field schema for classification

## 🐛 Debugging

### Check Logs
```bash
# Application logs will show in console
npm run start:dev

# Check Redis connection
redis-cli ping
```

### Common Issues:
1. **Redis Connection Error**: Make sure Redis is running on port 6379
2. **LLM API Errors**: Verify API keys in .env file
3. **File Upload Errors**: Check file size (max 10MB) and supported formats
4. **Job Processing Stuck**: Check BullMQ dashboard or Redis for job status

### Monitor Job Queue
You can add Bull Dashboard for monitoring:
```bash
npm install bull-board
```

## 📊 Expected Response Format

### Successful Processing Response:
```json
{
  "classification": {
    "is_expense": true,
    "expense_type": "meals",
    "language": "English",
    "language_confidence": 95
  },
  "extraction": {
    "supplier_name": "Restaurant ABC",
    "total_amount": "25.50 EUR",
    "transaction_date": "2024-01-15"
  },
  "compliance": {
    "validation_result": {
      "is_valid": true,
      "issues_count": 0,
      "issues": []
    }
  },
  "citations": {
    "citations": {},
    "metadata": {
      "total_fields_analyzed": 3,
      "average_confidence": 0.85
    }
  }
}
```

## 🔄 Next Steps

1. **Refine Prompts**: Refine prompts based on results
2. **Run tests on different countries**: 
3. **Concurrency test**: Upload more files (eg: 5) concurrently
