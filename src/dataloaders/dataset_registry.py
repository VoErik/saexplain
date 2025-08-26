"""Holds metadata about the supported datasets"""

INFO = {
    "HAM10000": {
        "description": """A benchmark dataset of over 10,000 dermatoscopic images of pigmented skin lesions across 
        seven diagnostic categories, intended for training automated classification models.""",
        "url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
        "task": "Image Classification, Image Segmentation",
        "modality": "Dermatoscopy",
        "size": 10015,
        "classes": {
            "akiec": "Actinic keratoses and intraepithelial carcinoma",
            "bcc": "Basal cell carcinoma",
            "bkl": "Benign keratosis-like lesions",
            "df": "Dermatofibroma",
            "mel": "Melanoma",
            "nv": "Melanocytic nevi",
            "vasc": "Vascular lesions"
        },
        "license": "CC-BY 4.0",
        "mean": [0.7635, 0.5462, 0.5706],
        "std": [0.1404, 0.1520, 0.1693]
    },
    "Fitzpatrick17k": {
        "description": """
        A dataset of 16,577 clinical images sourced from two dermatology atlases — DermaAmin and Atlas
        Dermatologico — with Fitzpatrick skin type labels. It is designed to audit and improve fairness in 
        dermatological models.
        """,
        "url": "https://github.com/mattgroh/fitzpatrick17k",
        "task": "Image Classification",
        "modality": "Clinical Photography",
        "size": 16577,
        "classes": "114 dermatologic conditions, plus Fitzpatrick skin type labels (I-VI).",
        "license": "CC-BY-NC-SA 3.0",
        "mean": [0.6174, 0.4769, 0.4240],
        "std": [0.2319, 0.2033, 0.2060]
    },

    "SCIN": {
        "description": """"
        The SCIN (Skin Condition Image Network) open access dataset aims to supplement publicly available dermatology 
        datasets from health system sources with representative images from internet users. To this end, the SCIN 
        dataset was collected from Google Search users in the United States through a voluntary, consented image 
        donation application. The SCIN dataset is intended for health education and research, and to increase the 
        diversity of dermatology images available for public use.
        The SCIN dataset contains 5,000+ volunteer contributions (10,000+ images) of common dermatology conditions. 
        Contributions include Images, self-reported demographic, history, and symptom information, and self-reported 
        Fitzpatrick skin type (sFST). In addition, dermatologist labels of the skin condition and estimated Fitzpatrick 
        skin type (eFST) and layperson estimated Monk Skin tone (eMST) labels are provided for each contribution.
        """,
        "url": "https://github.com/google-research-datasets/scin",
        "task": "Image Classification",
        "modality": "Self-reported Images",
        "size": 10380,
        "classes": "tbd",
        "license": "https://github.com/google-research-datasets/scin/blob/main/LICENSE",
        "mean": [0.5624, 0.4620, 0.4086],
        "std": [0.2599, 0.2255, 0.2182]
    },
    "MRA-MIDAS": {
        "description": """A prospectively-recruited dataset designed to mirror real-world clinical scenarios, containing 
        paired dermoscopic and clinical images of skin lesions with extensive histopathologic confirmation.""",
        "url": "https://doi.org/10.71718/15nz-jv40",
        "task": "Image Classification",
        "modality": "Multimodal (Dermoscopy & Clinical Photography)",
        "size": 3763,
        "classes": "Benign melanocytic nevi, melanomas (in-situ and invasive), and surgically-eligible intermediate melanocytic tumors.",
        "mean": [0.6319, 0.5169, 0.4673],
        "std": [0.2313, 0.2102, 0.2161]
    },
    "ClinicalPhotos": {
        "description": "Concatenation of SCIN, Fitzpatrick, and MRA-MIDAS",
        "mean": [0.6002, 0.4763, 0.4236],
        "std": [0.2435, 0.2126, 0.2122],
    },
    "DDI": {
        "description": "",
        "url": "https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965",
        "task": "Image Classification",
        "modality": "Dermatology Images",
        "size": 10380,
        "classes": "",
        "license": "Stanford University Dataset Research Use Agreement",
        "mean": [0.6012, 0.4859, 0.4055],
        "std": [0.2024, 0.2026, 0.2124]
    },
    "SKINCON-Fitzpatrick": "",
}