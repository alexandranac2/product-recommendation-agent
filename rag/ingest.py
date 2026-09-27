from langchain_core.documents import Document


def create_product_content(product):
    """
    Dynamically create content from ANY product structure.
    No hardcoded fields - works with any schema.
    """
    content_parts = []

    # Reserved keys that need special handling
    RESERVED_KEYS = {"id", "attributes"}

    # Add all top-level fields dynamically
    for key, value in product.items():
        if key in RESERVED_KEYS:
            continue

        # Format the key nicely
        readable_key = key.replace("_", " ").title()

        # Format the value based on field type
        if key == "price":
            formatted_value = f"${value}"
        elif key == "rating":
            formatted_value = f"{value}/5 stars"
        else:
            formatted_value = str(value)

        content_parts.append(f"{readable_key}: {formatted_value}")

    # Add attributes dynamically
    attrs = product.get("attributes", {})
    for key, value in attrs.items():
        readable_key = key.replace("_", " ").title()
        content_parts.append(f"{readable_key}: {value}")

    return " | ".join(content_parts)


def create_product_metadata(product):
    """
    Create metadata dictionary from a product dynamically.
    Copies all product fields and flattens attributes inline.

    Args:
        product: Dictionary containing product data

    Returns:
        Dictionary with product metadata (attributes flattened inline)
    """
    # Copy all product fields
    metadata = product.copy()

    # Extract and remove attributes to flatten them
    attributes = metadata.pop("attributes", {})

    # Flatten attributes inline with the rest of the metadata
    return {**metadata, **attributes}


def create_documents(products):
    """
    Convert products to Document objects.
    100% reusable - works with any product structure.
    """
    documents = []

    for product in products:
        content = create_product_content(product)

        doc = Document(
            page_content=content,
            metadata=create_product_metadata(product),
        )
        documents.append(doc)

    return documents
