from rag.ingest import create_product_content, create_product_metadata

PRODUCT = {
    "id": 7,
    "name": "Trail Shoe",
    "price": 120.0,
    "rating": 4.6,
    "attributes": {"terrain": "trail", "waterproof": "yes"},
}


def test_content_includes_formatted_fields_and_attributes():
    content = create_product_content(PRODUCT)
    assert "Name: Trail Shoe" in content
    assert "Price: $120.0" in content
    assert "Rating: 4.6/5 stars" in content
    assert "Terrain: trail" in content
    assert "Id:" not in content


def test_metadata_flattens_attributes_without_mutating_input():
    metadata = create_product_metadata(PRODUCT)
    assert metadata["terrain"] == "trail"
    assert "attributes" not in metadata
    assert "attributes" in PRODUCT


def test_every_catalogue_product_becomes_a_document(products):
    from rag.ingest import create_documents

    documents = create_documents(products)
    assert len(documents) == len(products)
    assert all(doc.metadata["id"] == p["id"] for doc, p in zip(documents, products))
