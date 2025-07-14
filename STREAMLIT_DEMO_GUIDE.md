# 🚀 Streamlit Demo Guide

## How to Run the Streamlit Demo

### 1. Start the Streamlit App
```bash
streamlit run streamlit_demo.py
```

### 2. Access the Demo
- The app will automatically open in your browser at `http://localhost:8501`
- If it doesn't open automatically, navigate to that URL manually

### 3. Using the Demo Interface

#### **Configuration (Sidebar)**
- **Country**: Select "Germany" or "Italy" for compliance rules
- **ICP**: Choose from "Global People", "goGlobal", "Parakar", "Atlas"
- **LlamaIndex API Key**: Enter your API key (pre-filled for demo)

#### **File Upload**
- Click "Choose expense files" to upload documents
- Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP, GIF, DOC, DOCX
- You can upload multiple files at once

#### **Processing**
- Click "🚀 Process Expenses" to start processing
- Watch real-time progress updates
- Results will appear automatically when complete

#### **Results View**
Each processed file shows:
- **🏷️ Classification**: Expense type, language, confidence scores
- **📋 Extraction**: Supplier info, amounts, line items
- **⚖️ Compliance**: Policy violations and recommendations
- **📄 Raw JSON**: Complete result data with download option

## 🎯 Demo Features

### **Automatic Processing**
- ✅ **File Upload**: Drag & drop or browse for files
- ✅ **Real-time Progress**: Live updates during processing
- ✅ **Multiple Files**: Process several documents at once
- ✅ **Auto Classification**: No manual input needed

### **Rich Results Display**
- ✅ **Tabbed Interface**: Organized result presentation
- ✅ **Metrics & Charts**: Visual data representation
- ✅ **Expandable Sections**: Detailed information on demand
- ✅ **JSON Download**: Export results for further use

### **Professional UI**
- ✅ **Responsive Design**: Works on desktop and mobile
- ✅ **Custom Styling**: Professional appearance
- ✅ **Error Handling**: Clear error messages
- ✅ **User Guidance**: Helpful tooltips and instructions

## 📊 Example Demo Flow

1. **Upload Files**: Add your expense receipts/invoices
2. **Configure**: Set country and ICP in sidebar
3. **Process**: Click the process button
4. **Review Results**: 
   - See automatic expense type classification
   - Review extracted data (supplier, amounts, dates)
   - Check compliance issues and recommendations
5. **Download**: Export JSON results if needed

## 🔧 Troubleshooting

### Common Issues:
- **"Please enter your LlamaIndex API key"**: Add your API key in the sidebar
- **Upload fails**: Ensure file format is supported
- **Processing hangs**: Check internet connection and API key validity
- **No results**: Verify uploaded files contain expense data

### Debug Mode:
The demo runs with `debug_mode=False` for cleaner output. To enable debug logging, modify line 89 in `streamlit_demo.py`:
```python
debug_mode=True  # Change from False to True
```

## 🎨 Customization

You can easily customize the demo:
- **Colors**: Modify CSS in the `st.markdown()` section
- **Layout**: Adjust column layouts and tab organization
- **Features**: Add new result visualizations or metrics
- **Configuration**: Add more countries or ICP options

## 💡 Demo Tips

- **Best Results**: Use clear, high-quality images of receipts
- **Multiple Languages**: The system handles German, English, and other languages
- **Various Types**: Try different expense types (meals, accommodation, travel, etc.)
- **Compliance**: Notice how different ICPs have different compliance rules

The Streamlit demo provides a professional, user-friendly interface to showcase the expense processing system's capabilities!
