import json
import csv
import os

# Funktion, um .ndjson in .csv zu konvertieren
def ndjson_to_csv(ndjson_file_name, csv_file_name):
    with open("GoogleDraw/"+ndjson_file_name, 'r') as jsonf, open(csv_file_name, 'w', newline='') as csvf:
        # Laden der ersten Zeile um die Feldnamen zu extrahieren
        first_line = json.loads(jsonf.readline())
        fieldnames = list(first_line.keys())
        jsonf.seek(0) # Zurück zum Anfang der Datei

        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for line in jsonf:
            data = json.loads(line)
            writer.writerow(data)

# Aktuelles Verzeichnis einholen
directory = os.getcwd()

# Alle .ndjson Dateien im aktuellen Verzeichnis finden
for filename in os.listdir(directory+"/GoogleDraw"):
    if filename.endswith(".ndjson"):
        # Neuen Dateinamen für die CSV-Datei erstellen
        csv_file_name = f"{os.path.splitext(filename)[0]}.csv"
        ndjson_to_csv(filename, csv_file_name)
        print(f"Converted {filename} to {csv_file_name}")
