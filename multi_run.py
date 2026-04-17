import subprocess
import sys
import os

PYTHON = sys.executable

# ==============================
# 1) Définition des profils
# ==============================

RUNS = [
    # DOL
    ("CH_DOL_H01", "CH_DOL_H01_2015-09-29_01","best"),
    ("CH_DOL_V01", "CH_DOL_V01_2016-08-11_01","best"),

    # CER
    ("IT_CER_CB1", "IT_CER_CB1_2013-08-16_01","best"),
    ("IT_CER_CB1", "IT_CER_CB1_2013-10-09_01","best"),
    ("IT_CER_CB1", "IT_CER_CB1_2014-09-14_01","best"),
    ("IT_CER_CB1", "IT_CER_CB1_2015-07-21_01","best"),
    ("IT_CER_CB1", "IT_CER_CB1_2015-08-20_01","best"),
    ("IT_CER_CB1", "IT_CER_CB1_2016-08-24_01","best"),
    ("IT_CER_CB1", "IT_CER_CB1_2018-08-23_01","best"),
    ("IT_CER_CB1", "IT_CER_CB1_2021-08-23_01","best"),

    # ATT
    ("CH_ATT_MV1", "CH_ATT_MV1_2022-10-07_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2021-09-24_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2020-09-22_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2019-09-17_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2018-10-15_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2017-09-25_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2016-09-23_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2014-09-30_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2010-07-14_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2008-11-10_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2008-07-16_01","all"),
    ("CH_ATT_MV1", "CH_ATT_MV1_2007-08-25_01","all"),

    # STT
    ("CH_STT_S07", "CH_STT_S07_2006-08-07_01","best"),
    ("CH_STT_S07", "CH_STT_S07_2007-08-11_01","best"),
    ("CH_STT_S07", "CH_STT_S07_2008-09-02_01","best"),
    ("CH_STT_S07", "CH_STT_S07_2012-08-18_01","best"),
    ("CH_STT_S07", "CH_STT_S07_2019-09-12_01","best"),

    # SON
    ("AT_SON_SM1", "AT_SON_SM1_2016-08-30_01","best"),
    ("AT_SON_SM1", "AT_SON_SM1_2017-06-26_01","best"),
    ("AT_SON_SM1", "AT_SON_SM1_2017-07-31_01","best"),
     ]




# ==============================
# 2) Multi-run avec gestion d'erreurs
# ==============================

def multi_run():
    results_ok = []
    results_fail = []

    # dossier de logs (optionnel, mais pratique)
    log_dir = "logs_multi"
    os.makedirs(log_dir, exist_ok=True)

    for profile_name, survey_name, keep in RUNS:
        print(f"\n=== RUN {profile_name} | {survey_name} | keep={keep} ===")
        safe_name = survey_name.replace("/", "_")
        log_path = os.path.join(log_dir, f"{safe_name}.log")
        cmd = [PYTHON, "main.py", profile_name, survey_name, keep]

        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
        )

        # on écrit tout dans un log
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("CMD: " + " ".join(cmd) + "\n\n")
            f.write("=== STDOUT ===\n")
            f.write(proc.stdout or "")
            f.write("\n=== STDERR ===\n")
            f.write(proc.stderr or "")

        if proc.returncode == 0:
            print(" -> OK")
            results_ok.append((profile_name, survey_name))
        else:
            print(" -> ÉCHEC (code", proc.returncode, ")")
            print("    (détails dans", log_path, ")")
            results_fail.append((profile_name, survey_name, proc.returncode))

    # résumé final
    print("\n===== RÉSUMÉ MULTI-RUN =====")
    print(f"Profils OK   : {len(results_ok)}")
    print(f"Profils FAIL : {len(results_fail)}")

    if results_fail:
        print("\nListe des échecs :")
        for prof, surv, code in results_fail:
            print(f" - {prof} | {surv} (returncode={code})")

    return results_ok, results_fail


if __name__ == "__main__":
    multi_run()
