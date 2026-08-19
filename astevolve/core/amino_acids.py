

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

HYDROPHOBIC = set("AILMFWVY")
CHARGED = set("KRDE")

HELIX_FAV = set("AEHILMQRK")
BETA_FAV = set("VIFYWT")


__all__ = [
    "AA",
    "AA_SET",
    "BETA_FAV",
    "CHARGED",
    "HELIX_FAV",
    "HYDROPHOBIC",
]
