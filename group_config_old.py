import os
from kinematic_object import Group

# ------------------------------------------------------------
# User-editable global settings
# ------------------------------------------------------------

VIDEO_DURATION = 7
DEFAULT_TRIAL_NUM = 20

DATA_FOLDER = r"C:\Users\agrawal-admin\Desktop\TibiaTarsusPlatformODLight-Wayne-2024-10-19"
LANDING_DATA_FOLDER = r"C:\Users\agrawal-admin\Desktop\LandingData"

# ------------------------------------------------------------
# Default key points and angles
# ------------------------------------------------------------

KEY_POINTS = [
    "L-wing", "L-wing-hinge", "R-wing", "R-wing-hinge", "abdomen-tip",
    "platform-tip", "L-platform-tip", "R-platform-tip", "platform-axis",
    "R-fBC", "R-fCT", "R-fFT", "R-fTT", "R-fLT",
    "R-mBC", "R-mCT", "R-mFT", "R-mTT", "R-mLT",
    "R-hBC", "R-hCT", "R-hFT", "R-hTT", "R-hLT",
    "L-fBC", "L-fCT", "L-fFT", "L-fTT", "L-fLT",
    "L-mBC", "L-mCT", "L-mFT", "L-mTT", "L-mLT",
    "L-hBC", "L-hCT", "L-hFT", "L-hTT", "L-hLT"
]

# ------------------------------------------------------------
# Easy places to change fly number / trial number
# ------------------------------------------------------------

# Use the full counts from your main script, not the temporary test counts.
FLY_NUM = {
    "WT_T1_CTF": 15,
    "WT_T1_TTa": 15,
    "WT_T2_CTF": 18,
    "WT_T2_TTa": 15,
    "WT_T3_CTF": 17,
    "WT_T3_TTa": 20,

    # KIR experiment
    "CSS-0039_T2_TiTa": 15,
    "CSS-0048_T2_TiTa": 17,
    "G106_T2_TTa": 16,
    "G107_T2_TTa": 14,
    "G108_T2_TTa": 17,
    "G114_T2_TTa": 16,
    "G115_T2_TTa": 16,
    "G116_T2_TTa": 18,
    "G117_T2_TTa": 18,
    "G118_T2_TTa": 15,
    "G119_T2_TTa": 18,

    # GTACR experiment
    "WT_Green": 9,
    "ANxGTACR": 12,
    "LexA_Br": 15,
    "MTGal4": 15,
    "IavxGTACR": 17,
    "CSS048xGTACR": 23,
    "CSS021xGTACR": 19,

    # CsChrimson Experiment: Low intensity
    "ADxChr-400uW": 15,
    "IavxChr-400uW": 15,
    "HP2xChr-400uW": 26,
    "TaCSxCHR-400uW": 5,
    "AllCSxChr-400uW": 15,

    # CsChrimson Experiment: Med intensity
    "IAVxCHR-4mW": 8,
    "HP2xCHR-4mW": 5,
    "TaBriLexAR-4mW": 9,
    "TaCSxCHR-4mW": 10,
    "CSS0048xCHR-4mW": 10,
    "BiCSxCHR-4mW": 5,
    "BiCS-HaltxCHR-4mW": 7,

    # CsChrimson: High intensity
    "IAVxCHR-12mW": 12,
    "HP2xChr-12mW": 5,
    "TaBriLexAR-12mW": 4,
    "TaCSxCHR-12mW": 4,
    "CSS0048xCHR-12mW": 18,
    "BiCS-HaltWgxCHR-12mW": 5,
    "BICSxCHR-12mW": 13,
    "BICSHALTxCHR-12mW": 5,
    "CSS0021xCHR-12mW": 10,

    # CsChrimson: AN
    "ANxCHR-400uW": 15,
    "ANxChr-4mW": 10,
    "ANxCHR-12mW": 10,
}

# Trial number per group.
# If a group is not listed here, DEFAULT_TRIAL_NUM is used.
TRIAL_NUM = {

    # GTACR experiment
    "WT_Green": 30,
    "ANxGTACR": 30,
    "LexA_Br": 30,
    "MTGal4": 30,
    "IavxGTACR": 30,
    "CSS048xGTACR": 30,
    "CSS021xGTACR": 30,

    # CsChrimson Experiment: Low intensity
    "ADxChr-400uW": 30,
    "IavxChr-400uW": 30,
    "HP2xChr-400uW": 30,
    "TaCSxCHR-400uW": 30,
    "AllCSxChr-400uW": 30,

    # CsChrimson Experiment: Med intensity
    "IAVxCHR-4mW": 30,
    "HP2xCHR-4mW": 30,
    "TaBriLexAR-4mW": 30,
    "TaCSxCHR-4mW": 30,
    "CSS0048xCHR-4mW": 30,
    "BiCSxCHR-4mW": 30,
    "BiCS-HaltxCHR-4mW": 30,

    # CsChrimson: High intensity
    "IAVxCHR-12mW": 30,
    "HP2xChr-12mW": 30,
    "TaBriLexAR-12mW": 30,
    "TaCSxCHR-12mW": 30,
    "CSS0048xCHR-12mW": 30,
    "BiCS-HaltWgxCHR-12mW": 30,
    "BICSxCHR-12mW": 30,
    "BICSHALTxCHR-12mW": 30,
    "CSS0021xCHR-12mW": 30,

    # CsChrimson: AN
    "ANxCHR-400uW": 30,
    "ANxChr-4mW": 30,
    "ANxCHR-12mW": 30,
}

# Use the full counts from your main script, not the temporary test counts.
TRIAL_OFFSET = {
    "WT_T1_CTF": 3,
    "WT_T1_TTa": 3,
    "WT_T2_CTF": 3,
    "WT_T2_TTa": 3,
    "WT_T3_CTF": 3,
    "WT_T3_TTa": 3,

    "CSS-0039_T2_TiTa": 3,
    "CSS-0048_T2_TiTa": 3,
    "G106_T2_TTa": 3,
    "G107_T2_TTa": 3,
    "G108_T2_TTa": 3,
    "G114_T2_TTa": 3,
    "G115_T2_TTa": 3,
    "G116_T2_TTa": 3,
    "G117_T2_TTa": 3,
    "G118_T2_TTa": 3,
    "G119_T2_TTa": 3,

    "WT_Green": 0,
    "ANxGTACR": 0,
    "LexA_Br": 0,
    "MTGal4": 0,
    "IavxGTACR": 0,
    "CSS048xGTACR": 0,
    "CSS021xGTACR": 0,

    # CsChrimson Experiment: Low intensity
    "ADxChr-400uW": 0,
    "IavxChr-400uW": 0,
    "HP2xChr-400uW": 0,
    "TaCSxCHR-400uW": 0,
    "AllCSxChr-400uW": 0,

    # CsChrimson Experiment: Med intensity
    "IAVxCHR-4mW": 0,
    "HP2xCHR-4mW": 0,
    "TaBriLexAR-4mW": 0,
    "TaCSxCHR-4mW": 0,
    "CSS0048xCHR-4mW": 0,
    "BiCSxCHR-4mW": 0,
    "BiCS-HaltxCHR-4mW": 0,

    # CsChrimson: High intensity
    "IAVxCHR-12mW": 0,
    "HP2xChr-12mW": 0,
    "TaBriLexAR-12mW": 0,
    "TaCSxCHR-12mW": 0,
    "CSS0048xCHR-12mW": 0,
    "BiCS-HaltWgxCHR-12mW": 0,
    "BICSxCHR-12mW": 0,
    "BICSHALTxCHR-12mW": 0,
    "CSS0021xCHR-12mW": 0,

    # CsChrimson: AN
    "ANxCHR-400uW": 0,
    "ANxChr-4mW": 0,
    "ANxCHR-12mW": 0,
}



# ------------------------------------------------------------
# FPS per group
# ------------------------------------------------------------

FPS = {
    "WT_T1_CTF": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "WT_T1_TTa": [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 250, 250, 250],
    "WT_T2_CTF": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "WT_T2_TTa": [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
    "WT_T3_CTF": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "WT_T3_TTa": [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 250, 250, 250, 250, 250],

    "CSS-0039_T2_TiTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "CSS-0048_T2_TiTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "G106_T2_TTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "G107_T2_TTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "G108_T2_TTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "G114_T2_TTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "G115_T2_TTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "G116_T2_TTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "G117_T2_TTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "G118_T2_TTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "G119_T2_TTa": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],

    "WT_Green": [250, 250, 250, 250, 250, 250, 250, 250, 250],
    "ANxGTACR": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "LexA_Br": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "MTGal4": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "IavxGTACR": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "CSS048xGTACR": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "CSS021xGTACR": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],

    # CsChrimson Experiment: Low intensity
    "ADxChr-400uW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "IavxChr-400uW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "HP2xChr-400uW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "TaCSxCHR-400uW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "AllCSxChr-400uW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],

    # CsChrimson Experiment: Med intensity
    "IAVxCHR-4mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "HP2xCHR-4mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "TaBriLexAR-4mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "TaCSxCHR-4mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "CSS0048xCHR-4mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "BiCSxCHR-4mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "BiCS-HaltxCHR-4mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],

    # CsChrimson: High intensity
    "IAVxCHR-12mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "HP2xChr-12mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "TaBriLexAR-12mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "TaCSxCHR-12mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "CSS0048xCHR-12mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "BiCS-HaltWgxCHR-12mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "BICSxCHR-12mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "BICSHALTxCHR-12mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "CSS0021xCHR-12mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],

    # CsChrimson: AN
    "ANxCHR-400uW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "ANxChr-4mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
    "ANxCHR-12mW": [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250],
}

# ------------------------------------------------------------
# Group path settings
# ------------------------------------------------------------

GROUP_INFO = {
    # WT experiment
    "WT_T1_CTF": {
        "group_name": "WT-T1-CxTr",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\LPAcrossLegsJoints\T1-CxTr"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T1-CxTr\T1-CxTr-LL_new.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T1-CxTr\T1-CxTr-MOC_new.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T1-CxTr\T1-CxTr-MOL_new.xlsx"),
    },
    "WT_T1_TTa": {
        "group_name": "WT-T1-TiTa",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\LPAcrossLegsJoints\T1-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T1-TiTa\T1-TiTa-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T1-TiTa\T1-TiTa-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T1-TiTa\T1-TiTa-MOL.xlsx"),
    },
    "WT_T2_CTF": {
        "group_name": "WT-T2-CxTr",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\LPAcrossLegsJoints\T2-CxTr"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T2-CxTr\T2-CxTr-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T2-CxTr\T2-CxTr-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T2-CxTr\T2-CxTr-MOL.xlsx"),
    },
    "WT_T2_TTa": {
        "group_name": "WT-T2-TiTa",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\LPAcrossLegsJoints\T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T2-TiTa\T2-TiTa-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T2-TiTa\T2-TiTa-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T2-TiTa\T2-TiTa-MOL.xlsx"),
    },
    "WT_T3_CTF": {
        "group_name": "WT-T3-CxTr",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\LPAcrossLegsJoints\T3-CxTr"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T3-CxTr\T3-CxTr-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T3-CxTr\T3-CxTr-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T3-CxTr\T3-CxTr-MOL.xlsx"),
    },
    "WT_T3_TTa": {
        "group_name": "WT-T3-TiTa",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\LPAcrossLegsJoints\T3-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T3-TiTa\T3-TiTa-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T3-TiTa\T3-TiTa-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"LPAcrossLegsJoints\T3-TiTa\T3-TiTa-MOL.xlsx"),
    },

    # KIR experiment
    "CSS-0039_T2_TiTa": {
        "group_name": "CSS-0039",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\HCS+_UASKir2.1eGFP\CSS-0039_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\CSS-0039_T2-TiTa\CSS-0039-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\CSS-0039_T2-TiTa\CSS-0039-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\CSS-0039_T2-TiTa\CSS-0039-MOL.xlsx"),
    },
    "CSS-0048_T2_TiTa": {
        "group_name": "CSS-0048",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-03-14\HCS+_UASKir2.1eGFP\CSS-0048_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\CSS-0048_T2-TiTa\CSS-0048-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\CSS-0048_T2-TiTa\CSS-0048-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\CSS-0048_T2-TiTa\CSS-0048-MOL.xlsx"),
    },
    "G106_T2_TTa": {
        "group_name": "G106-HP1",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-03-14\HCS+_UASKir2.1eGFP\G106-HP1_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G106-HP1_T2-TiTa\G106-HP1-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G106-HP1_T2-TiTa\G106-HP1-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G106-HP1_T2-TiTa\G106-HP1-MOL.xlsx"),
    },
    "G107_T2_TTa": {
        "group_name": "G107-HP2",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-05-30\HCS+_UASKir2.1eGFP\G107-HP2_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G107-HP2_T2-TiTa\G107-HP2-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G107-HP2_T2-TiTa\G107-HP2-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G107-HP2_T2-TiTa\G107-HP2-MOL.xlsx"),
    },
    "G108_T2_TTa": {
        "group_name": "G108-HP3",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-05-30\HCS+_UASKir2.1eGFP\G108-HP3_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G108-HP3_T2-TiTa\G108-HP3-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G108-HP3_T2-TiTa\G108-HP3-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G108-HP3_T2-TiTa\G108-HP3-MOL.xlsx"),
    },
    "G114_T2_TTa": {
        "group_name": "G114-ClFl",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-05-30\HCS+_UASKir2.1eGFP\G114-ClFl_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G114-ClFl_T2-TiTa\G114-CLFL-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G114-ClFl_T2-TiTa\G114-CLFL-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G114-ClFl_T2-TiTa\G114-CLFL-MOL.xlsx"),
    },
    "G115_T2_TTa": {
        "group_name": "G115-Iav",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-05-30\HCS+_UASKir2.1eGFP\G115-Iav_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G115-Iav_T2-TiTa\G115-IAV-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G115-Iav_T2-TiTa\G115-IAV-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G115-Iav_T2-TiTa\G115-IAV-MOL.xlsx"),
    },
    "G116_T2_TTa": {
        "group_name": "G116-ClEx",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-05-30\HCS+_UASKir2.1eGFP\G116-ClEx_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G116-ClEx_T2-TiTa\G116-CLEX-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G116-ClEx_T2-TiTa\G116-CLEX-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G116-ClEx_T2-TiTa\G116-CLEX-MOL.xlsx"),
    },
    "G117_T2_TTa": {
        "group_name": "G117-HkFl",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-05-30\HCS+_UASKir2.1eGFP\G117-HkFl_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G117-HkFl_T2-TiTa\G117-HKFL-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G117-HkFl_T2-TiTa\G117-HKFL-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G117-HkFl_T2-TiTa\G117-HKFL-MOL.xlsx"),
    },
    "G118_T2_TTa": {
        "group_name": "G118-HkEx",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-05-30\HCS+_UASKir2.1eGFP\G118-HkEx_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G118-HkEx_T2-TiTa\G118-HKEX-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G118-HkEx_T2-TiTa\G118-HKEX-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G118-HkEx_T2-TiTa\G118-HKEX-MOL.xlsx"),
    },
    "G119_T2_TTa": {
        "group_name": "G119-Club",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-05-30\HCS+_UASKir2.1eGFP\G119-Club_T2-TiTa"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G119-Club_T2-TiTa\G119-CLUB-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G119-Club_T2-TiTa\G119-CLUB-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"HCS+_UASKir2.1eGFP\G119-Club_T2-TiTa\G119-CLUB-MOL.xlsx"),
    },

    # GTACR experiment
    "WT_Green": {
        "group_name": "WT-Green",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\Optogenetics\WT-Green-Max"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\WT-Green\WT-Green-Max-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\WT-Green\WT-Green-Max-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\WT-Green\WT-Green-Max-MOL.xlsx"),
    },
    "ANxGTACR": {
        "group_name": "ANxGTACR",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\Optogenetics\ANxGTACR-Max"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\ANxGTACR-12mW\ANxGTACR-12mW-ALL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\ANxGTACR-12mW\ANxGTACR-12mW-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\ANxGTACR-12mW\ANxGTACR-12mW-MOL.xlsx"),
    },
    "LexA_Br": {
        "group_name": "LexA-Br-Green",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\TaBRIxLexAG-12mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\TaBRIxLexAG-12mW\TaBRIxLexAG-12mW-ALL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\TaBRIxLexAG-12mW\TaBRIxLexAG-12mW-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\TaBRIxLexAG-12mW\TaBRIxLexAG-12mW-MOL.xlsx"),
    },
    "MTGal4": {
        "group_name": "MTGal4",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\Optogenetics\GTACRxEmpty-Max"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\MTGal4xGTACR-12mW\MTGal4xGTACR-12mW-ALL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\MTGal4xGTACR-12mW\MTGal4xGTACR-12mW-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\MTGal4xGTACR-12mW\MTGal4xGTACR-12mW-MOL.xlsx"),
    },
    "IavxGTACR": {
        "group_name": "IavxGTACR",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\Optogenetics\IavxGTACR-Max"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\IAVxGTACR-12mW\IAVxGTACR-12mW-ALL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\IAVxGTACR-12mW\IAVxGTACR-12mW-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\IAVxGTACR-12mW\IAVxGTACR-12mW-MOL.xlsx"),
    },
    "CSS048xGTACR": {
        "group_name": "CSS048xGTACR",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\Optogenetics\CSS048xGTACR-Max"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\CS048xGTACR-12mW\CS048xGTACR-12mW-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\CS048xGTACR-12mW\CS048xGTACR-12mW-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\CS048xGTACR-12mW\CS048xGTACR-12mW-MOL.xlsx"),
    },
    "CSS021xGTACR": {
        "group_name": "CSS021xGTACR",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\Optogenetics\CSS021xGTACR"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\CSS021xGTACR-12mW\CSS021xGTACR-12mW-LL.xlsx"),
        "moc_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\CSS021xGTACR-12mW\CSS021xGTACR-12mW-MOC.xlsx"),
        "mol_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\GTACR\CSS021xGTACR-12mW\CSS021xGTACR-12mW-MOL.xlsx"),
    },

    # CsChrimson Experiment: Low intensity
    "ADxChr-400uW": {
        "group_name": "ADxCHR-400uW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\ADxCHR-400uW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Low Intensity\ADxCHR-400uW\ADxCHR-400uW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "IavxChr-400uW": {
        "group_name": "IavxChr-400uW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\IAVxCHR-400uW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Low Intensity\IavxChr-400uW\IavxChr-400uW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "HP2xChr-400uW": {
        "group_name": "HP2xCHR-400uW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\HP2xCHR-400uW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Low Intensity\HP2xChr-400uW\HP2xChr-400uW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "TaCSxCHR-400uW": {
        "group_name": "TaCSxCHR-400uW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\TaCSxChr-400uW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Low Intensity\TaCSxCHR-400uW\TaCSxCHR-400uW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "AllCSxChr-400uW": {
        "group_name": "ALLCSxCHR-400uW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\ALLCSxCHR-400uW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Low Intensity\ALLCSxCHR-400uW\ALLCSxCHR-400uW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },


    # CsChrimson Experiment: Med intensity
    "IAVxCHR-4mW": {
        "group_name": "IAVxCHR-4mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\IAVxCHR-4mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Medium intensity\IAVxCHR-4mW\IAVxCHR-4mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "HP2xCHR-4mW": {
        "group_name": "HP2xCHR-4mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\HP2xCHR-4mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Medium intensity\HP2xCHR-4mW\HP2xChr-4mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "TaBriLexAR-4mW": {
        "group_name": "TaBriLexAR-4mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\TaBRIxLexAR-4mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Medium intensity\TaBriLexAR-4mW\TaBRIxLexAR-4mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "TaCSxCHR-4mW": {
        "group_name": "TaCSxCHR-4mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\TaCSxCHR-4mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Medium intensity\TaCSxCHR-4mW\TaCSxChr-4mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "CSS0048xCHR-4mW": {
        "group_name": "CSS0048xCHR-4mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\CS048xCHR-4mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Medium intensity\CSS0048xCHR-4mW\CS048xCHR-4mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "BiCSxCHR-4mW": {
        "group_name": "BiCSxCHR-4mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\BICSxCHR-4mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER,  r"OPTO\CSChrimson\Medium intensity\BiCSxCHR-4mW\BiCSxChr-4mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "BiCS-HaltxCHR-4mW": {
        "group_name": "BiCS-HaltxCHR-4mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\BICSHALTxCHR-4mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\Medium intensity\BiCS-HaltxCHR-4mW\BiCS-HaltxChr-4mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },


    # CsChrimson: High intensity
    "IAVxCHR-12mW": {
        "group_name": "IAVxCHR-12mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\IAVxCHR-12mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\High intensity\IAVxCHR-12mW\IAVxCHR-12mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "HP2xChr-12mW": {
        "group_name": "HP2xChr-12mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\HP2xCHR-12mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\High intensity\HP2xChr-12mW\HP2xChr-12mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "TaBriLexAR-12mW": {
        "group_name": "TaBriLexAR-12mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\TaBRIxLexAR-12mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\High intensity\TaBriLexAR-12mW\TaBRIxLexAR-12mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "TaCSxCHR-12mW": {
        "group_name": "TaCSxCHR-12mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\TaCSxCHR-12mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\High intensity\TaCSxCHR-12mW\TaCSxCHR-12mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "CSS0048xCHR-12mW": {
        "group_name": "CSS0048xCHR-12mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\CS048xCHR-12mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\High intensity\CSS0048xCHR-12mW\CS048xCHR-12mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "BiCS-HaltWgxCHR-12mW": {
        "group_name": "BiCS-HaltWgxCHR-12mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\BICSHALTWGxCHR-12mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\High intensity\BiCS-HaltWgxCHR-12mW\BiCS-HaltWgxChr-12mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "BICSxCHR-12mW": {
        "group_name": "BICSxCHR-12mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-07-04\Optogenetics\BiCSxChr-8mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\High intensity\BICSxCHR-12mW\BICSxCHR-12mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "BICSHALTxCHR-12mW": {
        "group_name": "BICSHALTxCHR-12mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\BICSHALTxCHR-12mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\High intensity\BICSHALTxCHR-12mW\BICSHALTxCHR-12mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "CSS0021xCHR-12mW": {
        "group_name": "CSS0021xCHR-12mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\CSS021xCHR-12mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\High intensity\CSS0021xCHR-12mW\CS021xCHR-12mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },



    # CsChrimson: AN
    "ANxCHR-400uW": {
        "group_name": "ANxCHR-400uW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\ANxCHR-400uW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\AN\ANxCHR-400uW\ANxCHR-400uW-ALL.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "ANxChr-4mW": {
        "group_name": "ANxChr-4mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\ANxCHR-4mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\AN\ANxCHR-4mW\ANxCHR-4mW.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
    "ANxCHR-12mW": {
        "group_name": "ANxCHR-12mW",
        "kine_path": os.path.join(DATA_FOLDER, r"Network-01-18-2026\OPTO\ANxCHR-12mW"),
        "ll_path": os.path.join(LANDING_DATA_FOLDER, r"OPTO\CSChrimson\AN\ANxCHR-12mW\ANxCHR-12mW-ALL.xlsx"),
        "moc_path": "NoPath",
        "mol_path": "NoPath",
    },
}

# ------------------------------------------------------------
# Builders
# ------------------------------------------------------------

def get_trial_num(group_key):
    if group_key in TRIAL_NUM:
        return TRIAL_NUM[group_key]
    return DEFAULT_TRIAL_NUM


def build_one_group(group_key):
    info = GROUP_INFO[group_key]

    group = Group(
        moc_data_path=info["moc_path"],
        mol_data_path=info["mol_path"],
        ll_data_path=info["ll_path"],
        fly_kinematic_data_path=info["kine_path"],
        group_name=info["group_name"],
        joints=KEY_POINTS,
        total_fly_number=FLY_NUM[group_key],
        fps=FPS[group_key],
        trial_num=get_trial_num(group_key),
        trials_offset=TRIAL_OFFSET[group_key],
        video_duration=VIDEO_DURATION
    )

    return group


def group_paths_available(group_key, require_kinematics=False):
    info = GROUP_INFO[group_key]

    paths_to_check = [
        info["ll_path"],
        info["moc_path"],
        info["mol_path"],
    ]
    if require_kinematics:
        paths_to_check.insert(0, info["kine_path"])

    for path in paths_to_check:
        if path == "NoPath":
            continue
        if not os.path.exists(path):
            return False

    return True


def build_groups(group_keys=None, skip_missing=True, require_kinematics=False):
    groups = dict()

    if group_keys is None:
        group_keys = list(GROUP_INFO.keys())

    for group_key in group_keys:
        if not group_paths_available(group_key, require_kinematics=require_kinematics):
            if skip_missing:
                print(f"Skipping unavailable group: {group_key}")
                continue
            raise FileNotFoundError(
                f"Configured paths for group '{group_key}' are not all available."
            )

        groups[group_key] = build_one_group(group_key)

    return groups
