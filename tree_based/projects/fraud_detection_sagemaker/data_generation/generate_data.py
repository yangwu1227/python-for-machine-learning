from faker import Faker
from typing import List, Tuple, Dict, Union, Optional
import datetime
import itertools
import numpy as np
import pandas as pd

Faker.seed(0)
np.random.seed(0)

# Number of unique credit card numbers
NUM_UNIQUE_CCS = 40 * 10**3
# Transaction start date
START_TRANS_DATE = datetime.datetime(2012, 1, 15)
# Transaction end date
END_TRANS_DATE = datetime.datetime(2012, 3, 15)


def gen_fraud_data(
    num_unique_ccs: int = NUM_UNIQUE_CCS,
    start_trans_date: datetime.datetime = START_TRANS_DATE,
    end_trans_date: datetime.datetime = END_TRANS_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic fraud data set.

    Parameters
    ----------
    num_unique_ccs : int, optional
        Number of valid credit card numbers, by default NUM_UNIQUE_CCS
    start_trans_date : datetime.datetime, optional
        Transaction start date, by default START_TRANS_DATE
    end_trans_date : datetime.datetime, optional
        Transaction end date, by default END_TRANS_DATE

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Transactions and identity tables as pandas DataFrames.
    """
    # -------------------------- Generate synthetic data ------------------------- #

    # Instantiate Faker instance
    fake = Faker()
    cc_nums = [
        fake.credit_card_number() for _ in range(num_unique_ccs)
    ]  # Generate a valid and unique credit card numbers

    cc_types = [
        fake.credit_card_provider() for _ in range(num_unique_ccs)
    ]  # Generate credit card provider names

    num_trans_per_cc = np.ceil(
        np.random.exponential(scale=3, size=num_unique_ccs)
    ).astype(
        np.int32
    )  # Generate numbers of transactions per credit card number, the overall distribution follows an exponential distribution with most credit card numbers having fewer transactions

    # Generate random IPv4 addresses or network with a valid CIDR
    cc_ipv4 = [fake.ipv4() for _ in range(num_unique_ccs)]

    cc_phone_number = [
        fake.phone_number() for _ in range(num_unique_ccs)
    ]  # Generate random phone numbers

    # Generate random mobile subscriber identification numbers (MSISDNs)
    cc_device_id = [fake.msisdn() for _ in range(num_unique_ccs)]

    # ------------------------ Generate transactions table ----------------------- #

    # Generate transactions table
    data = {
        # Generate random UUIDs (Universally Unique Identifiers)
        "TransactionID": [fake.uuid4() for _ in range(sum(num_trans_per_cc))],
        "TransactionDT": [
            fake.date_time_between_dates(
                datetime_start=start_trans_date, datetime_end=end_trans_date
            )
            for _ in range(sum(num_trans_per_cc))
        ],  # Generate random dates and times between two given dates
        # For each credit card number, generate the same number of transactions as the number of transactions per credit card number
        "card_no": list(
            itertools.chain.from_iterable(
                [
                    [cc_num] * num_trans
                    for cc_num, num_trans in zip(cc_nums, num_trans_per_cc)
                ]
            )
        ),
        # For each credit card type, generate the same number of transactions as the number of transactions per credit card number
        "card_type": list(
            itertools.chain.from_iterable(
                [
                    [card] * num_trans
                    for card, num_trans in zip(cc_types, num_trans_per_cc)
                ]
            )
        ),
        # Generate random email addresses and extract the domain name
        "email_domain": [
            fake.ascii_email().split("@")[1] for _ in range(sum(num_trans_per_cc))
        ],
        # Generate random product codes
        "ProductCD": np.random.choice(
            ["45", "AB", "L", "Y", "T"], size=sum(num_trans_per_cc)
        ),
        # Generate random transaction amounts
        "TransactionAmt": np.abs(
            np.ceil(np.random.exponential(scale=10, size=sum(num_trans_per_cc)) * 100)
        ).astype(np.int32),
    }
    transactions = pd.DataFrame(data).sort_values(by=["TransactionDT"])

    # -------------------------- Generate identity table ------------------------- #

    # If you want to make the # of observations in the identity table less than that in the transactions table which may be more realistic in a practical scenario, change the size argument below
    # In this case, we will make it such that the # of observations in the identity table is 80% of that in the transactions table
    identity_transactions_idx = np.random.choice(
        a=transactions.shape[0], size=int(transactions.shape[0] * 0.8), replace=False
    )  # Generate random indices for the identity table

    # Each credit card number has the same number of transactions as the number of transactions per credit card number, so we can use the same number of transactions to generate the same number of rows in the identity table
    id_data = {
        "IpAddress": list(
            itertools.chain.from_iterable(
                [
                    [ipv4] * num_trans
                    for ipv4, num_trans in zip(cc_ipv4, num_trans_per_cc)
                ]
            )
        ),
        "PhoneNo": list(
            itertools.chain.from_iterable(
                [
                    [phone_num] * num_trans
                    for phone_num, num_trans in zip(cc_phone_number, num_trans_per_cc)
                ]
            )
        ),
        "DeviceID": list(
            itertools.chain.from_iterable(
                [
                    [device_id] * num_trans
                    for device_id, num_trans in zip(cc_device_id, num_trans_per_cc)
                ]
            )
        ),
    }
    identity = pd.DataFrame(id_data)
    # Add a column 'TransactionID' to the identity table
    identity["TransactionID"] = transactions.TransactionID
    assert identity.shape[0] == transactions.shape[0]

    # Subset the identity table using the random indices generated above, where the # of observations in the identity table is 80% of that in the transactions table
    identity = identity.loc[identity_transactions_idx]
    # Reset the index of the identity table
    identity.reset_index(drop=True, inplace=True)
    identity = identity[["TransactionID", "IpAddress", "PhoneNo", "DeviceID"]]

    # Join two tables for the convenience of generating label column 'isFraud'
    # If 'identity' has fewer rows than 'transactions', then the columns in 'identity' will lead to missing values in the joined table, which has the same number of rows as 'transactions'
    # This is realistic since we may not have the identity information for all transactions
    full_two_df = transactions[
        [
            "TransactionID",
            "card_no",
            "card_type",
            "email_domain",
            "ProductCD",
            "TransactionAmt",
        ]
    ].merge(identity, on="TransactionID", how="left")

    # --------------------- Generate target column 'isFraud' --------------------- #

    is_fraud = []

    for idx, row in full_two_df.iterrows():
        # A row of data in the joined table
        (
            card_no,
            card_type,
            email,
            product_type,
            transcation_amount,
            ip_address,
            phone_no,
            device_id,
        ) = (
            str(row["card_no"]),
            row["card_type"],
            row["email_domain"],
            row["ProductCD"],
            row["TransactionAmt"],
            str(row["IpAddress"]),
            str(row["PhoneNo"]),
            str(row["DeviceID"]),
        )

        # Rules for generating the target column 'isFraud' (1 or 0)
        if email in ["hotmail.com", "gmail.com", "yahoo.com"]:
            if product_type in ["45"]:
                # If the product type is 45 and the email domain is in the list above, then the probability of fraud is high
                is_fraud.append(int(np.random.uniform() < 0.9))
            else:
                # If the device ID is not missing and ends with either 16, 78, or 23, then the probability of fraud is low
                if (device_id != "nan") and (
                    device_id.endswith("16")
                    or device_id.endswith("78")
                    or device_id.endswith("23")
                ):
                    is_fraud.append(int(np.random.uniform() < 0.1))
                else:
                    # If the device id is missing or does not end with either 16, 78, or 23, then the probability of fraud is low
                    is_fraud.append(int(np.random.uniform() < 0.05))
        else:
            # If the email domain is not in the list above and the transaction amount is greater than 3000, then the probability of fraud is high
            if transcation_amount > 3000:
                is_fraud.append(int(np.random.uniform() < 0.8))
            else:
                # About 35,000 observations are in these categories
                if card_type in [
                    "Diners Club / Carte Blanche",
                    "JCB 15 digit",
                    "Maestro",
                ]:
                    if (
                        card_no.endswith("001")
                        or card_no.endswith("002")
                        or card_no.endswith("003")
                        or card_no.endswith("004")
                        or card_no.endswith("005")
                        or card_no.endswith("007")
                        or card_no.endswith("008")
                        or card_no.endswith("009")
                    ) or (
                        (phone_no != "nan")
                        and (
                            phone_no.endswith(".227")
                            or phone_no.endswith(".104")
                            or phone_no.endswith(".251")
                            or phone_no.endswith(".181")
                        )
                    ):
                        is_fraud.append(int(np.random.uniform() < 0.3))
                    else:
                        if (ip_address != "nan") and (
                            ip_address.endswith(".227")
                            or ip_address.endswith(".104")
                            or ip_address.endswith(".251")
                            or ip_address.endswith(".181")
                        ):
                            is_fraud.append(int(np.random.uniform() < 0.2))
                        else:
                            is_fraud.append(int(np.random.uniform() < 0.1))
                else:
                    is_fraud.append(int(np.random.uniform() < 0.0001))

    # Print the fraud ratio
    print("fraud ratio", sum(is_fraud) / len(is_fraud))

    transactions["isFraud"] = is_fraud
    return transactions, identity


if __name__ == "__main__":
    transaction, identity = gen_fraud_data()
    transaction.to_csv("raw_data/transaction.csv", index=False)
    identity.to_csv("raw_data/identity.csv", index=False)
