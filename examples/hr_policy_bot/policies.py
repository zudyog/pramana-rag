"""HR policy catalog for a fictional company, Nimbus Labs.

Entirely invented for this example — not sourced from any real company or
document. Exists to prove that pramana's abstractions (ConcernChunker,
Retriever, GroundedAnswerer) generalize past the fashion-e-commerce domain
in examples/shopbot/, using deliberately different concern names
(summary/eligibility/procedure/exceptions/faq instead of
identity/variants/policy/care/faq) to avoid the appearance of a relabeled
copy.
"""

POLICIES = [
    {
        "id": "pol001",
        "name": "Remote Work Policy",
        "summary": "Employees may work remotely on a full-time or hybrid "
        "basis, subject to manager approval and role requirements.",
        "eligibility": "All full-time employees who have completed their "
        "90-day probation period. Roles requiring on-site equipment access "
        "(hardware lab, manufacturing floor) are not eligible for full-time "
        "remote status.",
        "procedure": "Submit a remote work request through the HR portal "
        "at least 2 weeks before the desired start date. Your manager must "
        "approve the request, and it is reviewed annually.",
        "exceptions": "Employees on a performance improvement plan (PIP) "
        "must work on-site until the plan concludes.",
        "faqs": [
            {
                "question": "Can I work remotely from another country?",
                "answer": "International remote work requires separate "
                "approval from HR and Legal due to tax and employment law "
                "implications. Submit a request at least 6 weeks in advance.",
            },
        ],
    },
    {
        "id": "pol002",
        "name": "Parental Leave Policy",
        "summary": "Nimbus Labs provides 16 weeks of paid parental leave "
        "for the birth, adoption, or fostering of a child.",
        "eligibility": "All employees, regardless of gender, who have been "
        "employed for at least 6 months at the start of leave.",
        "procedure": "Notify your manager and HR at least 30 days before "
        "the expected leave start date where possible. Leave can be taken "
        "continuously or split into two blocks within the first 12 months.",
        "faqs": [
            {
                "question": "Can I extend my leave unpaid?",
                "answer": "Yes, employees may request up to 8 additional "
                "weeks of unpaid leave, subject to manager approval.",
            },
            {
                "question": "Does parental leave affect my PTO accrual?",
                "answer": "No, PTO continues to accrue normally during "
                "paid parental leave.",
            },
        ],
    },
    {
        "id": "pol003",
        "name": "Expense Reimbursement Policy",
        "summary": "Employees are reimbursed for reasonable, "
        "business-related expenses incurred while performing their job.",
        "eligibility": "All employees. Contractors should refer to their "
        "individual contract terms instead of this policy.",
        "procedure": "Submit receipts through the Expenses portal within "
        "30 days of the purchase. Expenses over $500 require pre-approval "
        "from your manager.",
        "exceptions": "Alcohol, personal entertainment, and first-class "
        "travel are not reimbursable under any circumstance.",
        "faqs": [
            {
                "question": "What happens if I submit a receipt after 30 days?",
                "answer": "Late submissions are reviewed on a case-by-case "
                "basis by Finance and are not guaranteed to be reimbursed.",
            },
        ],
    },
    {
        "id": "pol004",
        "name": "Paid Time Off (PTO) Policy",
        "summary": "Full-time employees accrue 20 days of PTO per year, "
        "prorated for the first year of employment.",
        "eligibility": "All full-time employees. Part-time employees "
        "accrue PTO on a prorated basis according to hours worked.",
        "procedure": "Request PTO through the HR portal at least 5 "
        "business days in advance for planned leave. Manager approval is "
        "required.",
    },
    {
        "id": "pol005",
        "name": "Equipment and Device Policy",
        "summary": "Nimbus Labs provides a laptop and standard peripherals "
        "to every employee upon joining.",
        "eligibility": "All employees, from their first day.",
        "procedure": "IT ships equipment before your start date for "
        "remote employees, or has it ready at your desk for on-site "
        "employees.",
        "exceptions": "Employees in hardware engineering roles receive "
        "additional specialized equipment, requested separately through "
        "the Hardware Requests form.",
    },
]
