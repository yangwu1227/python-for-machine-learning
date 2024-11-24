import numpy as np
import pandas as pd

# --------------------------- Datatype transformer --------------------------- #


def DtypeTransformer(data: pd.DataFrame, copy=False) -> pd.DataFrame:
    """
    Function to transform the data types of the dataframe.
    """
    if copy:
        data = data.copy(deep=True)

    # Enforce columns order so downstream transformers can expect consistency
    data = data[
        [
            # Id columns
            "encounter_id",
            "patient_nbr",
            # Numerical columns
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
            # Arbitrary categorical columns
            "race",
            "gender",
            "age",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "payer_code",
            "medical_specialty",
            "diag_1",
            "diag_2",
            "diag_3",
            "max_glu_serum",
            "A1Cresult",
            "change",
            "diabetesMed",
            # Medication categorical columns
            "metformin",
            "repaglinide",
            "nateglinide",
            "chlorpropamide",
            "glimepiride",
            "acetohexamide",
            "glipizide",
            "glyburide",
            "tolbutamide",
            "pioglitazone",
            "rosiglitazone",
            "acarbose",
            "miglitol",
            "troglitazone",
            "tolazamide",
            "examide",
            "citoglipton",
            "insulin",
            "glyburide-metformin",
            "glipizide-metformin",
            "glimepiride-pioglitazone",
            "metformin-rosiglitazone",
            "metformin-pioglitazone",
        ]
    ]

    # Integer id's can be downcast to 32 bit integers
    ids_map = {"encounter_id": np.int32, "patient_nbr": np.int32}
    data = data.astype(ids_map)

    # Numerical features can be downcast to 16 bit integers (max 32767)
    num_col_map = {
        "time_in_hospital": np.int16,
        "num_lab_procedures": np.int16,
        "num_procedures": np.int16,
        "num_medications": np.int16,
        "number_outpatient": np.int16,
        "number_emergency": np.int16,
        "number_inpatient": np.int16,
        "number_diagnoses": np.int16,
    }
    data = data.astype(num_col_map)

    # Convert integer-based categorical features to 'object' in preparation for downstream transformer to work them to 'category'
    cat_col_map = {
        "admission_type_id": "object",
        "discharge_disposition_id": "object",
        "admission_source_id": "object",
    }
    data = data.astype(cat_col_map)

    return data


# ---------------------- Restore column order and names ---------------------- #


def RestoreTransformer(data: np.ndarray) -> pd.DataFrame:
    """
    Take numpy array returned by upstream transformer and restore to DataFrame with correct column order and names.

    Parameters
    ----------
    data : np.ndarray
        Numpy array returned by upstream transformer.

    Returns
    -------
    pd.DataFrame
        DataFrame with correct column order and names.
    """
    # Assuming the numpy array is returned by a ColumnTransformer (must be the case to get correct column order)
    # The first 23 columns are the medication columns since ColumnTransformer appends the 'pass through' columns on the left
    data = pd.DataFrame(
        data,
        columns=[
            # Medication categorical columns
            "metformin",
            "repaglinide",
            "nateglinide",
            "chlorpropamide",
            "glimepiride",
            "acetohexamide",
            "glipizide",
            "glyburide",
            "tolbutamide",
            "pioglitazone",
            "rosiglitazone",
            "acarbose",
            "miglitol",
            "troglitazone",
            "tolazamide",
            "examide",
            "citoglipton",
            "insulin",
            "glyburide-metformin",
            "glipizide-metformin",
            "glimepiride-pioglitazone",
            "metformin-rosiglitazone",
            "metformin-pioglitazone",
            # Passed-through columns from ColumnTransformer
            # Id columns
            "encounter_id",
            "patient_nbr",
            # Numerical columns
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
            # Arbitrary categorical columns
            "race",
            "gender",
            "age",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "payer_code",
            "medical_specialty",
            "diag_1",
            "diag_2",
            "diag_3",
            "max_glu_serum",
            "A1Cresult",
            "change",
            "diabetesMed",
        ],
    )

    return data


# ------------------ Convert columns to categorical datatype ----------------- #


def CatTransformer(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function to convert columns to categorical datatype.
    """
    # Convert categorical features to 'category' datatype
    cat_col = [
        "race",
        "gender",
        "age",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "payer_code",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
        "max_glu_serum",
        "A1Cresult",
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
        "change",
        "diabetesMed",
    ]

    for col in cat_col:
        data[col] = data[col].astype("category")

    return data


# ------------------ Feature engineer for numerical columns ------------------ #


def NumTransformer(data: pd.DataFrame) -> pd.DataFrame:
    """
    Group by aggregation of numerical columns based on patient id and demographic characteristics.
    """
    # Define list of numerical columns to aggregate
    num_col = [
        "num_medications",
        "num_procedures",
        "time_in_hospital",
        "number_emergency",
        "number_inpatient",
        "number_outpatient",
        "num_lab_procedures",
        "number_diagnoses",
    ]

    for group in ["age", "gender", "race", "patient_nbr", "medical_specialty"]:
        for agg_func in ["mean", "median", "max", "sum"]:
            data[[col + f"_{agg_func}_by_{group}" for col in num_col]] = data.groupby(
                group
            )[num_col].transform(agg_func, engine="cython")

    return data
