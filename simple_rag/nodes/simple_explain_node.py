from typing import Any, Dict, List

from simple_rag.agent.simple_state import SimpleAgentState
from simple_rag.pricing import extract_price_range


def _extract_numeric_field(products: List[Dict[str, Any]], field_name: str) -> List[float]:
    """Dynamically extract numeric field from products if it exists"""
    values = []
    for product in products:
        value = product.get(field_name)
        if value is not None:
            try:
                values.append(float(value))
            except (ValueError, TypeError):
                pass
    return values


def _extract_text_field(products: List[Dict[str, Any]], field_name: str) -> List[str]:
    """Dynamically extract text field from products if it exists"""
    values = []
    for product in products:
        value = product.get(field_name)
        if value and isinstance(value, (str, int, float)):
            values.append(str(value))
    return values


def _find_common_fields(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dynamically discover what fields exist across products"""
    if not products:
        return {}
    
    # Get all unique keys from all products
    all_keys = set()
    for product in products:
        all_keys.update(product.keys())
    
    field_info = {}
    
    # Analyze each field
    for key in all_keys:
        # Skip internal/metadata fields
        if key in {'id', 'score', 'score_type', 'content'}:
            continue
        
        # Check if field exists in all/most products
        present_count = sum(1 for p in products if key in p and p[key] is not None)
        
        if present_count > 0:
            # Sample values to determine type
            sample_values = [p.get(key) for p in products[:3] if key in p and p[key] is not None]
            
            # Determine field type
            is_numeric = all(
                isinstance(v, (int, float)) or 
                (isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit())
                for v in sample_values if v is not None
            )
            
            field_info[key] = {
                'present_count': present_count,
                'is_numeric': is_numeric,
                'sample_values': sample_values[:3]
            }
    
    return field_info


def simple_explain_node(state: SimpleAgentState) -> SimpleAgentState:
    """
    Generate explanation based on actual product data - NO LLM, NO HALLUCINATIONS
    Works dynamically with ANY product structure - no hardcoded fields
    """
    recommendations = state["recommendations"]
    query = state["query"]

    if not recommendations:
        state["explanation"] = "No products found matching your criteria."
        return state

    # Same price bound the refine step filtered on
    _, max_price = extract_price_range(query)
    max_price = None if max_price == float("inf") else max_price

    # Analyze actual product data dynamically
    top_products = recommendations[:3]
    
    # Dynamically discover what fields exist
    field_info = _find_common_fields(top_products)
    
    # Build explanation from actual data
    explanation_parts = []
    
    # Basic count
    explanation_parts.append(f"Found {len(recommendations)} products matching your search.")
    
    # Price information (if price field exists)
    if 'price' in field_info:
        prices = _extract_numeric_field(top_products, 'price')
        if prices:
            avg_price = sum(prices) / len(prices)
            if max_price:
                explanation_parts.append(
                    f"All products are priced under ${max_price:.0f} "
                    f"(average: ${avg_price:.1f})."
                )
            else:
                explanation_parts.append(f"Average price: ${avg_price:.1f}.")
    
    # Rating/quality indicators (dynamically find rating-like fields)
    rating_fields = [k for k in field_info.keys() 
                    if 'rating' in k.lower() or 'score' in k.lower() or 'quality' in k.lower()]
    for rating_field in rating_fields[:1]:  # Use first rating field found
        ratings = _extract_numeric_field(top_products, rating_field)
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            field_label = rating_field.replace('_', ' ').title()
            if avg_rating >= 4.5:
                explanation_parts.append(
                    f"Excellent {field_label.lower()} (average {avg_rating:.1f})."
                )
            elif avg_rating >= 4.0:
                explanation_parts.append(
                    f"Good {field_label.lower()} (average {avg_rating:.1f})."
                )
    
    # Category/type information (dynamically find category-like fields)
    category_fields = [k for k in field_info.keys() 
                      if 'categor' in k.lower() or 'type' in k.lower() or 'kind' in k.lower()]
    for category_field in category_fields[:1]:  # Use first category field found
        categories = _extract_text_field(top_products, category_field)
        if categories:
            unique_categories = list(set(categories))
            field_label = category_field.replace('_', ' ').title()
            if len(unique_categories) == 1:
                explanation_parts.append(
                    f"All products are {unique_categories[0]} {field_label.lower()}."
                )
            elif len(unique_categories) <= 3:
                explanation_parts.append(
                    f"Products include: {', '.join(unique_categories[:3])}."
                )
    
    # Use case/purpose (dynamically find use-case-like fields)
    use_case_fields = [k for k in field_info.keys() 
                      if 'use' in k.lower() or 'purpose' in k.lower() or 'for' in k.lower()]
    for use_case_field in use_case_fields[:1]:  # Use first use-case field found
        use_cases = _extract_text_field(top_products, use_case_field)
        if use_cases:
            unique_use_cases = [uc for uc in set(use_cases) if uc]
            if unique_use_cases:
                explanation_parts.append(
                    f"Suitable for: {', '.join(unique_use_cases[:2])}."
                )
    
    # Top product highlight (use 'name' or first text field)
    if top_products:
        top_product = top_products[0]
        name_field = 'name' if 'name' in top_product else (
            next((k for k in top_product.keys() 
                 if k not in {'id', 'price', 'rating', 'score'} and 
                    isinstance(top_product[k], str)), None)
        )
        
        if name_field:
            top_name = top_product.get(name_field, 'product')
            price = top_product.get('price', 0)
            if price:
                explanation_parts.append(
                    f"Top recommendation: {top_name} at ${price:.2f}."
                )
            else:
                explanation_parts.append(f"Top recommendation: {top_name}.")
    
    # Combine into final explanation
    explanation = " ".join(explanation_parts)
    if not explanation or explanation == f"Found {len(recommendations)} products matching your search.":
        explanation = f"Found {len(recommendations)} products matching your search criteria."
    
    state["explanation"] = explanation
    return state
