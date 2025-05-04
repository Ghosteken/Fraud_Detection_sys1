from faker import Faker
import pandas as pd
import random

fake = Faker()

def generate_fake_transaction():
    return {
        "transaction_id": fake.uuid4(),
        "buyer_name": fake.name(),
        "seller_name": fake.name(),
        "property_type": random.choice(["Residential", "Commercial", "Industrial", "Land"]),
        "property_value": round(random.uniform(50000, 500000), 2),
        "mortgage_amount": round(random.uniform(10000, 300000), 2),
        "latitude": fake.latitude(),
        "longitude": fake.longitude(),
        "buyer_latitude": fake.latitude(),
        "buyer_longitude": fake.longitude(),
        "transaction_month": random.randint(1, 12),
        "buyer_gender": random.choice(["Male", "Female"]),
        "ssn": fake.ssn()[5:],  # Just use last 4 digits
        "fraudulent": random.choice([0, 1])  # Add fraudulent column with random choice between 0 (legitimate) and 1 (fraudulent)
    }

def generate_transactions(num_records):
    transactions = []
    for _ in range(num_records):
        transactions.append(generate_fake_transaction())
    return transactions

def save_transactions_to_csv(num_records, filename="transactions_log.csv"):
    transactions = generate_transactions(num_records)
    df = pd.DataFrame(transactions)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    save_transactions_to_csv(1000)  # Generate 1000 transactions
    print("Transactions saved to transactions_log.csv")
