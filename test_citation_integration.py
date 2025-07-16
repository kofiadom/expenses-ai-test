#!/usr/bin/env python3
"""
Test script for integrated citation generation.
"""

import json
from pathlib import Path
from citation_generator import generate_citations, get_citation_stats

def test_citation_integration():
    """Test the integrated citation generation with sample data."""
    
    print("🧪 Testing Integrated Citation Generation")
    print("=" * 50)
    
    # Sample structured output (typical extraction result)
    structured_output = {
        "supplier_name": "THE SUSHI CLUB",
        "total_amount": 54.46,
        "date_of_issue": "2019-02-05",
        "currency": "EUR",
        "contact_phone": "+49 30 23 916 036",
        "contact_email": "info@thesushiclub.de",
        "line_items": [
            {"description": "Miso Soup", "quantity": 1, "unit_price": 3.90, "total_price": 3.90},
            {"description": "Salmon Sashimi", "quantity": 2, "unit_price": 12.50, "total_price": 25.00}
        ]
    }
    
    # Sample extraction requirements
    extraction_requirements = json.dumps({
        "file_related_requirements": [
            {
                "field_type": "Supplier Name",
                "rule": "Extract the name of the business or supplier",
                "description": "The name of the business that issued the receipt"
            },
            {
                "field_type": "Total Amount", 
                "rule": "Extract the total amount paid",
                "description": "The final total amount on the receipt"
            },
            {
                "field_type": "Date of Issue",
                "rule": "Extract the date when the receipt was issued", 
                "description": "The date the transaction occurred"
            }
        ]
    })
    
    # Sample markdown content
    markdown_content = """
# Restaurant Receipt

**THE SUSHI CLUB**
Mohrenstr.42, 10117 Berlin
Tel: +49 30 23 916 036
Email: info@thesushiclub.de
Website: WWW.TheSushiClub.de

---

**Date:** 05.02.2019
**Time:** 23:10:54
**Table:** 24

## Order Details

| Item | Qty | Price | Total |
|------|-----|-------|-------|
| Miso Soup | 1 | €3.90 | €3.90 |
| Salmon Sashimi | 2 | €12.50 | €25.00 |

---

**Subtotal:** €42.40
**Tax (19%):** €8.06
**Service Charge:** €4.00
**Total:** €54.46

**Payment Method:** Cash

Thank you for dining with us!
    """
    
    # Test filename
    test_filename = "test_receipt"
    
    print(f"📝 Testing citation generation for: {test_filename}")
    print(f"📊 Structured output fields: {len(structured_output)}")
    print(f"📄 Markdown length: {len(markdown_content)} characters")
    
    # Generate citations
    try:
        citations = generate_citations(
            structured_output=structured_output,
            extraction_requirements=extraction_requirements,
            markdown_content=markdown_content,
            filename=test_filename
        )
        
        print("\n✅ Citation generation successful!")
        
        # Get and display statistics
        stats = get_citation_stats(citations)
        print(f"\n📈 Citation Statistics:")
        print(f"   - Total fields: {stats.get('total_fields', 0)}")
        print(f"   - Field citations: {stats.get('fields_with_field_citations', 0)}")
        print(f"   - Value citations: {stats.get('fields_with_value_citations', 0)}")
        print(f"   - Field citation rate: {stats.get('field_citation_rate', 0):.1%}")
        print(f"   - Value citation rate: {stats.get('value_citation_rate', 0):.1%}")
        print(f"   - Average confidence: {stats.get('average_confidence', 0):.2f}")
        
        # Show sample citations
        if "citations" in citations:
            print(f"\n📍 Sample Citations:")
            for field_name, field_citations in list(citations["citations"].items())[:3]:
                print(f"\n   • {field_name}:")
                
                if "field_citation" in field_citations:
                    fc = field_citations["field_citation"]
                    print(f"     Field: '{fc.get('source_text', 'N/A')}' (confidence: {fc.get('confidence', 0):.1f})")
                
                if "value_citation" in field_citations:
                    vc = field_citations["value_citation"]
                    print(f"     Value: '{vc.get('source_text', 'N/A')}' (confidence: {vc.get('confidence', 0):.1f})")
        
        # Check if file was created
        citation_file = Path("citation_folder") / f"{test_filename}_citation.json"
        if citation_file.exists():
            print(f"\n📁 Citation file created: {citation_file}")
            print(f"   File size: {citation_file.stat().st_size} bytes")
        else:
            print(f"\n❌ Citation file not found: {citation_file}")
            
    except Exception as e:
        print(f"\n❌ Citation generation failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Citation integration test completed!")
    return True


if __name__ == "__main__":
    test_citation_integration()
