"""zUdyog Fashion's catalog, as used in RAG Essentials Book 1 (chapters 4-9).

Five products. p001 and p003 carry the full structured data given in the
book's Chapter 5 (identity, variants, policy, care, FAQs). p002, p004, and
p005 only carry the fields the book actually specifies for them (identity,
variants, and — where the book states it directly — care or a verified FAQ)
rather than inventing policy/care/FAQ detail the source material never gave.
"""

PRODUCTS = [
    {
        "id": "p001",
        "name": "Breathable Cotton Kurta",
        "description": "Lightweight daily-wear kurta crafted from 100% breathable cotton. "
        "Ideal for summer and warm weather.",
        "category": "kurta",
        "fabric": "100% cotton",
        "occasion": "casual, daily wear",
        "colors": ["white", "sky blue", "mint green"],
        "sizes": ["S", "M", "L", "XL", "XXL"],
        "price": 1299,
        "return_policy": "Returns accepted within 7 days. Item must be unworn "
        "and in original packaging with tags.",
        "exchange_policy": "One free size exchange within 10 days of delivery.",
        "care_instructions": "Machine wash cold on gentle cycle. Do not bleach. "
        "Tumble dry low. Iron on medium heat.",
        "faqs": [
            {
                "question": "Does this kurta shrink after washing?",
                "answer": "Minimal shrinkage of 2-3% may occur after first wash. "
                "We recommend washing cold and air drying to maintain fit.",
            },
            {
                "question": "Is this suitable for office wear?",
                "answer": "Yes, the cotton kurta is appropriate for casual office "
                "environments. Pair with straight-fit trousers for a polished look.",
            },
        ],
    },
    {
        "id": "p002",
        "name": "Heavyweight Woolen Shawl",
        "description": "Heavyweight woolen shawl, warm, ideal for winter evenings.",
        "category": "shawl",
        "occasion": "winter evenings",
        "colors": ["charcoal grey", "maroon"],
        "sizes": ["one size"],
        "price": 2199,
        "care_instructions": "Dry clean only.",
    },
    {
        "id": "p003",
        "name": "Printed Silk Saree",
        "description": "Six-yard printed silk saree with hand-embroidered zari border. "
        "Rich festive drape suitable for weddings, receptions, and formal celebrations.",
        "category": "saree",
        "fabric": "pure silk with zari embroidery",
        "occasion": "festive, wedding, formal",
        "colors": ["red", "gold", "emerald"],
        "sizes": ["standard — fits most"],
        "price": 4299,
        "return_policy": "Returns accepted within 7 days. Saree must be unstitched, "
        "unused, with original packaging.",
        "exchange_policy": "Exchanges available for colour only within 5 days.",
        "care_instructions": "Dry clean only. Machine washing will damage the zari border "
        "and cause silk to lose its sheen. Store in a muslin cloth bag away from "
        "direct sunlight.",
        "faqs": [
            {
                "question": "Can this be worn for a wedding?",
                "answer": "Yes. The zari border and silk fabric make this saree "
                "appropriate for weddings, receptions, and formal festivals.",
            },
            {
                "question": "Is the saree pre-stitched or unstitched?",
                "answer": "The saree is unstitched — a standard six-yard drape. "
                "No blouse is included. Blouse fabric can be sourced separately.",
            },
        ],
    },
    {
        "id": "p004",
        "name": "Linen Co-ord Set",
        "description": "Linen co-ord set, relaxed fit, summer casual.",
        "category": "coord-set",
        "occasion": "summer casual",
        "colors": ["beige", "sage", "dusty rose"],
        "sizes": ["XS", "S", "M", "L", "XL"],
        "price": 1899,
    },
    {
        "id": "p005",
        "name": "Embroidered Anarkali Suit",
        "description": "Embroidered anarkali suit with dupatta.",
        "category": "suit",
        "occasion": "festive",
        "colors": ["teal"],
        "sizes": ["S", "M", "L", "XL", "XXL"],
        "price": 3499,
        "faqs": [
            {
                "question": "Does this suit come with a dupatta?",
                "answer": "Yes, the anarkali suit comes with a matching "
                "embroidered dupatta in teal.",
            },
        ],
    },
]
