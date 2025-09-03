#!/usr/bin/env python3

import csv
import time
import requests
import re
from bs4 import BeautifulSoup
from rdkit import Chem
from dimorphite_dl import protonate_smiles


def get_smiles_from_hmdb(url):
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    th = soup.find("th", string=re.compile(r"^\s*SMILES\s*$", re.I))
    if th:
        td = th.find_next_sibling("td")
        if td:
            text = td.get_text(strip=True)
            if text and text.lower() != "not available":
                return text

    for dt in soup.find_all("dt"):
        if "smiles" in dt.get_text(strip=True).lower():
            dd = dt.find_next_sibling("dd")
            if dd:
                text = dd.get_text(strip=True)
                if text and text.lower() != "not available":
                    return text

    return None


def main():
    infile  = r"C:\Users\david.moore\OneDrive - Thermo Fisher Scientific\Desktop\Cluster\MordredVersion\UI\oxford.csv"
    outfile = "analytes_deprot.csv"

    with open(infile, newline="", encoding="utf-8") as fin, \
         open(outfile, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["Name", "HMDB_URL", "SMILES", "Deprot_SMILES"])

        for row in reader:
            name = row["Name"]
            url  = row["HMDB_URL"]
            print(f"⏳ Fetching SMILES for {name}…", end="", flush=True)

            smi = None
            try:
                smi = get_smiles_from_hmdb(url)
            except Exception as e:
                print(f" ❌ HTTP/error: {e}")

            if not smi:
                print(" – no SMILES found")
                writer.writerow([name, url, "", ""])
            else:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    print(" – invalid SMILES!")
                    writer.writerow([name, url, smi, ""])
                else:
                    # Enumerate all ionization microstates across full pH range (0–14)
                    try:
                        variants = protonate_smiles(
                            smi,
                            ph_min=12.5,
                            ph_max=12.5,
                            precision=0.5
                        )
                    except Exception as e:
                        print(f" ❌ protonate error: {e}")
                        variants = []

                    # Compute formal charge for each variant
                    charged = []
                    for v in variants:
                        m_var = Chem.MolFromSmiles(v)
                        if m_var is None:
                            continue
                        charge = Chem.GetFormalCharge(m_var)
                        charged.append((v, charge))

                    # Pick the most negatively charged (fully deprotonated) variant
                    deprot_smi = ""
                    if charged:
                        deprot_smi = min(charged, key=lambda x: x[1])[0]

                    writer.writerow([name, url, smi, deprot_smi])
                    print(f" ✔ {smi} → {deprot_smi}")

            time.sleep(0.1)

    print(f"\n✅ Done! See results in {outfile}")


if __name__ == "__main__":
    main()
